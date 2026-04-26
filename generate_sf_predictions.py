"""
Generate per-position Stockfish top-k predictions for a dataset.

Reads a dataset in all_d1 format (CSV or folder with train/eval/test CSVs),
evaluates each black-move position with Stockfish, and writes a new CSV.

Output schema (compound key: id + turn_index):
    id                    — game id
    turn_index            — ply index of the position within the game
    sf_d{D}_t{K}          — space-separated top-K Stockfish moves at depth D
    sf_d{D}_t{K}_status   — 1 if actual next move is in top-K list, else 0
    sf_ms_d{D}            — Stockfish inference time in milliseconds for depth D

Usage:
    # Whole folder, multiple depths and k values:
    python generate_sf_predictions.py \
        --input  data/all_d1 \
        --output data/all_d1_sf \
        --stockfish_path stockfish/stockfish-linux \
        --depth 5 10 --k 1 3 5

    # Single CSV:
    python generate_sf_predictions.py \
        --input  data/all_d1/test.csv \
        --output data/all_d1_sf/test.csv \
        --stockfish_path stockfish/stockfish-linux \
        --depth 10 --k 1 3 5
"""

import argparse
import re
import time
from pathlib import Path

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm


BLACK_MOVE_INDICES = list(range(11, 35, 2))


def add_p_to_pawn(moves):
    out = []
    for m in moves:
        if m in {"O-O", "O-O-O"}:
            out.append(m)
        elif m and m[0] not in {"N", "B", "Q", "R", "K", "O"}:
            out.append("P" + m if not m.startswith("P") else m)
        else:
            out.append(m)
    return out


def replay_board(tokens, prompt_end):
    """Return a chess.Board after replaying tokens[:prompt_end], or None on error."""
    board = chess.Board()
    for tok in tokens[:prompt_end]:
        clean = re.sub(r"[?!]", "", tok)
        if clean.startswith("P") and clean not in {"O-O", "O-O-O"}:
            clean = clean[1:]
        try:
            board.push_san(clean)
        except Exception:
            return None
    return board


def stockfish_at_depth(board, engine, max_k, depth):
    """Run Stockfish on board at given depth with multipv=max_k. Returns (moves_list, sf_ms)."""
    t0 = time.time()
    try:
        analysis = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=max_k)
    except Exception:
        return None, 0.0
    sf_ms = (time.time() - t0) * 1000

    if isinstance(analysis, dict):
        analysis = [analysis]

    moves = []
    for info in sorted(analysis, key=lambda x: x.get("multipv", 1)):
        pv = info.get("pv", [])
        if pv:
            moves.append(board.san(pv[0]))

    return add_p_to_pawn(moves), round(sf_ms, 2)


def normalize_move(move):
    return re.sub(r"[?!+#]", "", move).strip()


def process_csv(csv_in, csv_out, engine, ks, depths):
    df = pd.read_csv(csv_in)
    df = df[(df["victory_status"] == "mate") & (df["turns"] > 35)]
    max_k = max(ks)
    rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=Path(csv_in).name):
        tokens = str(getattr(row, "moves", "")).strip().split()
        game_id = getattr(row, "id", None)

        for turn_index in BLACK_MOVE_INDICES:
            if turn_index >= len(tokens):
                break
            board = replay_board(tokens, turn_index)
            if board is None:
                continue
            actual = normalize_move(tokens[turn_index])

            rec = {"id": game_id, "turn_index": turn_index}
            for depth in depths:
                moves_list, sf_ms = stockfish_at_depth(board, engine, max_k, depth)
                if moves_list is None:
                    continue
                for k in ks:
                    col = f"sf_d{depth}_t{k}"
                    top_k = moves_list[:k]
                    rec[col] = " ".join(top_k)
                    rec[f"{col}_status"] = 1 if actual in [normalize_move(m) for m in top_k] else 0
                    rec[f"{col}_ms"] = sf_ms

            if len(rec) > 2:  # has at least one depth result
                rows.append(rec)

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_out, index=False)
    print(f"Saved {len(out_df)} rows -> {csv_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="CSV file or folder with train/eval/test CSVs.")
    parser.add_argument("--output", required=True,
                        help="Output CSV (if input is a file) or output folder (if input is a folder).")
    parser.add_argument("--stockfish_path", required=True)
    parser.add_argument("-d", "--depth", type=int, nargs="+", default=[10])
    parser.add_argument("-k", "--k", type=int, nargs="+", default=[5])
    args = parser.parse_args()

    if not Path(args.stockfish_path).exists():
        raise FileNotFoundError(f"Stockfish not found: {args.stockfish_path}")

    depths = sorted(set(args.depth))
    ks = sorted(set(args.k))
    print(f"Stockfish depths={depths} top-ks={ks}")
    with chess.engine.SimpleEngine.popen_uci(args.stockfish_path) as engine:
        inp = Path(args.input)
        if inp.is_dir():
            for split in ["train", "eval", "test"]:
                csv_in = inp / f"{split}.csv"
                if csv_in.exists():
                    process_csv(csv_in, Path(args.output) / f"{split}.csv", engine, ks, depths)
        else:
            process_csv(args.input, args.output, engine, ks, depths)


if __name__ == "__main__":
    main()
