"""
Compare LLM predictions (from eval_detail.csv) against Stockfish top-k.
Uses precomputed Stockfish predictions from the dataset when available,
and only runs live Stockfish for positions that don't have them.

Usage:
    python compare_results.py \
        --eval_csv results/eval/all_d1/Qwen3.5-9B/eval_detail.csv \
        --stockfish_path /path/to/stockfish \
        --k 1 3 5 \
        --depth 10 \
        --output_dir results/eval/all_d1/Qwen3.5-9B/

    # Force live Stockfish for all positions (ignore precomputed):
    python compare_results.py ... --force_recompute
"""

import argparse
import os
import re
import sys
import time
import logging

import chess
import chess.engine
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def normalize_label(text):
    if not text:
        return ""
    text = re.sub(r"[?!]", "", str(text)).strip()
    return text.split()[0] if text else ""


def unique_in_order(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def get_board_after_moves(move_str):
    board = chess.Board()
    for tok in str(move_str).strip().split():
        if tok[0].isdigit() and tok.endswith("."):
            continue
        try:
            board.push_san(re.sub(r"[?!]", "", tok))
        except Exception:
            break
    return board


def parse_precomputed(precomputed_sf, max_k):
    """Parse comma-separated precomputed predictions into a list of up to max_k moves."""
    if not precomputed_sf or not isinstance(precomputed_sf, str) or not precomputed_sf.strip():
        return []
    moves = []
    for m in precomputed_sf.strip().split(","):
        norm = normalize_label(m)
        if norm and norm not in moves:
            moves.append(norm)
        if len(moves) >= max_k:
            break
    return moves


def get_stockfish_topk(prompt_moves, engine, k=5, depth=10):
    board = get_board_after_moves(prompt_moves)
    try:
        analysis = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=max(1, k))
    except Exception as e:
        log.warning("Stockfish analysis failed: %s", e)
        return []
    if isinstance(analysis, dict):
        analysis = [analysis]
    preds = []
    for info in sorted(analysis, key=lambda x: x.get("multipv", 1)):
        pv = info.get("pv", [])
        if pv:
            preds.append(normalize_label(board.san(pv[0])))
    return unique_in_order(preds)


def main():
    parser = argparse.ArgumentParser(description="Compare LLM predictions vs Stockfish.")
    parser.add_argument("--eval_csv", required=True,
                        help="eval_detail.csv saved by run_model.py test mode.")
    parser.add_argument("--stockfish_path", default=None,
                        help="Path to Stockfish binary (required when positions lack precomputed predictions or --force_recompute is set).")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--depth", type=int, default=10, help="Stockfish search depth for live runs.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same folder as eval_csv).")
    parser.add_argument("--max_positions", type=int, default=0, help="0 = all.")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore precomputed predictions and run live Stockfish for every position.")
    args = parser.parse_args()

    df = pd.read_csv(args.eval_csv)
    if args.max_positions:
        df = df.head(args.max_positions)
    log.info("Loaded %d positions from %s", len(df), args.eval_csv)

    ordered_k = tuple(sorted(set(args.k)))
    max_k = max(ordered_k)

    if args.force_recompute:
        n_live = len(df)
        log.info("--force_recompute: running live Stockfish for all %d positions.", n_live)
    else:
        has_precomputed = df["precomputed_sf"].notna() & (df["precomputed_sf"].astype(str).str.strip() != "")
        n_live = (~has_precomputed).sum()
        log.info("Precomputed available: %d | Needs live Stockfish: %d", has_precomputed.sum(), n_live)

    if n_live > 0 and not args.stockfish_path:
        log.error("%d positions need live Stockfish but --stockfish_path not provided.", n_live)
        sys.exit(1)

    def run_comparison(engine):
        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            actual = normalize_label(row.get("actual_label", ""))
            precomputed = row.get("precomputed_sf", "")

            sf_preds = [] if args.force_recompute else parse_precomputed(precomputed, max_k)
            sf_ms = 0.0

            if len(sf_preds) < max_k and engine is not None:
                t0 = time.time()
                sf_preds = get_stockfish_topk(row.get("prompt", ""), engine, k=max_k, depth=args.depth)
                sf_ms = (time.time() - t0) * 1000

            rec = dict(row)
            rec["stockfish_inference_ms"] = sf_ms
            rec["stockfish_source"] = f"live_depth{args.depth}" if sf_ms > 0 else "precomputed"
            for k in ordered_k:
                rec[f"stockfish_hit@{k}"] = actual in sf_preds[:k]
                rec[f"stockfish_top_{k}"] = ", ".join(sf_preds[:k])
            rows.append(rec)

            if (i + 1) % 10 == 0:
                log.info("  %d/%d positions compared", i + 1, len(df))

        return pd.DataFrame(rows)

    if n_live > 0:
        with chess.engine.SimpleEngine.popen_uci(args.stockfish_path) as engine:
            detail_df = run_comparison(engine)
    else:
        detail_df = run_comparison(engine=None)

    summary_rows = []
    for k in ordered_k:
        row = {"k": k, "positions": len(detail_df)}
        if f"llm_hit@{k}" in detail_df.columns:
            row["llm_pass@k"] = detail_df[f"llm_hit@{k}"].mean()
        if f"stockfish_hit@{k}" in detail_df.columns:
            row["stockfish_pass@k"] = detail_df[f"stockfish_hit@{k}"].mean()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    log.info("\n%s\nCOMPARISON SUMMARY\n%s\n%s",
             "=" * 60, summary_df.round(4).to_string(index=False), "=" * 60)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.eval_csv))
    os.makedirs(out_dir, exist_ok=True)
    detail_df.to_csv(os.path.join(out_dir, "comparison_detail.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    log.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()