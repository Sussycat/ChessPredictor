"""
Match rows between two eval_detail CSVs using (prompt, actual_label) as the join key.

Usage:
    python match_eval_results.py \
        --left  data/test1/eval_detail.csv \
        --right data/test2/eval_detail_2.csv \
        --output data/matched_eval.csv
"""

import argparse
import re
import chess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

JOIN_COLS = ["prompt", "actual_label"]


def legal_moves_at(full_moves_str, turn_index):
    """Replay full_moves_str up to turn_index tokens and return legal moves."""
    board = chess.Board()
    tokens = str(full_moves_str).strip().split()
    for tok in tokens[:turn_index]:
        tok = re.sub(r"[?!]", "", tok)
        if tok.startswith("P") and tok not in {"O-O", "O-O-O"}:
            tok = tok[1:]
        try:
            board.push_san(tok)
        except Exception:
            return ""
    moves = []
    for m in board.legal_moves:
        san = board.san(m)
        if san not in {"O-O", "O-O-O"} and san[0] not in {"N", "B", "Q", "R", "K"}:
            san = "P" + san
        moves.append(san)
    return " ".join(moves)


def load(path):
    df = pd.read_csv(path)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    for col in JOIN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left",   required=True, help="First eval_detail CSV (test1).")
    parser.add_argument("--right",  required=True, help="Second eval_detail CSV (test2).")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--how", default="inner",
                        choices=["inner", "left", "right", "outer"],
                        help="Join type (default: inner).")
    parser.add_argument("--test_csv", default=None,
                        help="Path to test.csv used to simulate game positions for legal moves.")
    parser.add_argument("--sf_d5", default=None,
                        help="Path to all_d1_sf_d5.csv to merge Stockfish depth-5 predictions.")
    parser.add_argument("--sf_d10", default=None,
                        help="Path to all_d1_sf_d10.csv to merge Stockfish depth-10 predictions.")
    parser.add_argument("--sf_d15", default=None,
                        help="Path to all_d1_sf_d15.csv to merge Stockfish depth-15 predictions.")
    args = parser.parse_args()

    left  = load(args.left)
    right = load(args.right)
    print(f"Left  rows: {len(left)}")
    print(f"Right rows: {len(right)}")

    left_dupe_idx  = left[left.duplicated(subset=JOIN_COLS, keep=False)].index.tolist()
    right_dupe_idx = right[right.duplicated(subset=["id", "turn_index"], keep=False)].index.tolist()
    if left_dupe_idx:
        print(f"Left  duplicate row numbers ({len(left_dupe_idx)}): {left_dupe_idx}")
    else:
        print("Left:  no duplicates")
    if right_dupe_idx:
        print(f"Right duplicate row numbers ({len(right_dupe_idx)}): {right_dupe_idx}")
    else:
        print("Right: no duplicates")
    right = right.drop_duplicates(subset=["id", "turn_index"])

    left["_left_idx"]   = left.index
    right["_right_idx"] = right.index

    left_rename  = {c: f"{c}_t1" for c in left.columns  if c not in JOIN_COLS + ["_left_idx"]}
    right_rename = {c: f"{c}_t2" for c in right.columns if c not in JOIN_COLS + ["_right_idx"]}

    def proximity_match(ldf, rdf, how="inner"):
        m = pd.merge(ldf, rdf, on=JOIN_COLS, how=how)
        m["_dist"] = (m["_left_idx"] - m["_right_idx"]).abs()
        return (
            m.sort_values("_dist")
            .drop_duplicates(subset=["_left_idx"], keep="first")
            .drop_duplicates(subset=["_right_idx"], keep="first")
        )

    lrenamed = left.rename(columns=left_rename)
    rrenamed = right.rename(columns=right_rename)

    # Pass 1
    pass1 = proximity_match(lrenamed, rrenamed, how=args.how)
    matched_left  = set(pass1["_left_idx"])
    matched_right = set(pass1["_right_idx"])

    # Pass 2: retry unmatched rows from both sides
    unmatched_left  = lrenamed[~lrenamed["_left_idx"].isin(matched_left)]
    unmatched_right = rrenamed[~rrenamed["_right_idx"].isin(matched_right)]
    print(f"Pass 1 matched: {len(pass1)} | Unmatched left: {len(unmatched_left)} | Unmatched right: {len(unmatched_right)}")

    if not unmatched_left.empty and not unmatched_right.empty:
        pass2 = proximity_match(unmatched_left, unmatched_right, how=args.how)
        print(f"Pass 2 matched: {len(pass2)}")
        combined = pd.concat([pass1, pass2], ignore_index=True)
    else:
        combined = pass1

    # Sort: game order is determined by the first left_idx seen per game,
    # then within each game sort by turn_index_t2 to enforce increasing order.
    combined = combined.copy()
    combined["_game_order"] = combined.groupby("id_t2")["_left_idx"].transform("min")
    merged = (
        combined
        .sort_values(["_game_order", "turn_index_t2"])
        .drop(columns=["_left_idx", "_right_idx", "_dist", "_game_order"])
        .reset_index(drop=True)
    )

    # Drop left id, keep right id and turn_index
    drop_cols = [c for c in [
        "id_t1", "precomputed_sf_t1",
        "llm_hit@1_t2", "llm_top_1_t2",
        "llm_hit@3_t2", "llm_top_3_t2",
        "llm_hit@5_t2", "llm_top_5_t2",
    ] if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    # Merge inference times: use right if left > 400, else left
    if "llm_inference_ms_t1" in merged.columns and "llm_inference_ms_t2" in merged.columns:
        merged["llm_inference_ms"] = merged.apply(
            lambda r: r["llm_inference_ms_t2"] if r["llm_inference_ms_t1"] > 400 else r["llm_inference_ms_t1"],
            axis=1,
        )
        merged = merged.drop(columns=["llm_inference_ms_t1", "llm_inference_ms_t2"])

    # Merge Stockfish predictions on (id_t2, turn_index_t2)
    for sf_path, label in [(args.sf_d5, "sf_d5"), (args.sf_d10, "sf_d10"), (args.sf_d15, "sf_d15")]:
        if sf_path:
            sf = pd.read_csv(sf_path)
            sf = sf.rename(columns={"id": "id_t2", "turn_index": "turn_index_t2"})
            sf["id_t2"] = sf["id_t2"].astype(str)
            merged["id_t2"] = merged["id_t2"].astype(str)
            merged["turn_index_t2"] = merged["turn_index_t2"].astype(int)
            sf["turn_index_t2"] = sf["turn_index_t2"].astype(int)
            merged = merged.merge(sf, on=["id_t2", "turn_index_t2"], how="left")
            print(f"Merged {label}: {sf_path}")

    # Add legal moves using test.csv game simulation
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv, usecols=["id", "moves", "black_rating", "white_rating"])
        test_df["id"] = test_df["id"].astype(str)
        game_moves  = dict(zip(test_df["id"], test_df["moves"].astype(str)))
        black_elo   = dict(zip(test_df["id"], test_df["black_rating"]))
        white_elo   = dict(zip(test_df["id"], test_df["white_rating"]))
        merged["id_t2"] = merged["id_t2"].astype(str)
        merged["black_elo"] = merged["id_t2"].map(black_elo)
        merged["white_elo"] = merged["id_t2"].map(white_elo)
        tqdm.pandas(desc="Computing legal moves")
        merged["legal_moves"] = merged.progress_apply(
            lambda r: legal_moves_at(game_moves.get(str(r["id_t2"]), ""), int(r["turn_index_t2"])),
            axis=1,
        )
    else:
        print("Skipping legal moves and elo (no --test_csv provided)")

    # Move id_t2 and turn_index_t2 to the front
    front = [c for c in ["id_t2", "turn_index_t2"] if c in merged.columns]
    merged = merged[front + [c for c in merged.columns if c not in front]]

    # Detect turn index anomalies per game: sequence must start at 11 and have
    # no gaps in the middle (step of 2). Missing at the tail is fine.
    missing_report = []
    for game_id, group in merged.groupby("id_t2", sort=False):
        indices = group["turn_index_t2"].dropna().astype(int).tolist()
        if not indices:
            continue
        issues = []
        if indices[0] != 11:
            issues.append(f"starts at {indices[0]} not 11")
        not_increasing = [indices[i] for i in range(1, len(indices)) if indices[i] <= indices[i-1]]
        if not_increasing:
            issues.append(f"not increasing at {not_increasing}")
        gaps = [indices[i] for i in range(1, len(indices)) if indices[i] - indices[i-1] != 2 and indices[i] > indices[i-1]]
        if gaps:
            issues.append(f"gaps before {gaps}")
        if issues:
            missing_report.append((game_id, issues))
    if missing_report:
        print(f"\nTurn index anomalies ({len(missing_report)} games):")
        for game_id, issues in missing_report:
            print(f"  {game_id}: {', '.join(issues)}")
    else:
        print("All games have valid turn index sequences.")

    print(f"Matched rows:    {len(merged)}")
    if args.how == "inner":
        print(f"Unmatched left:  {len(left)  - len(merged)}")
        print(f"Unmatched right: {len(right) - len(merged)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
