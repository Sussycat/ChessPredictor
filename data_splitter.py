import argparse
import pandas as pd
import chess
import chess.engine
import re
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

# ==========================================
# 1. CHESS LOGIC HELPERS
# ==========================================
def apply_pawn_prefix(san_move):
    if san_move in {"O-O", "O-O-O"}: return san_move
    if san_move[0] not in {"N", "B", "Q", "R", "K", "O"}:
        return "P" + san_move if not san_move.startswith("P") else san_move
    return san_move

def strip_pawn_prefix(move_str):
    if move_str.startswith("P") and move_str not in {"O-O", "O-O-O"}: return move_str[1:]
    return move_str

def clean_and_format_moves(movetext):
    """Strips clock data, numbers, and applies your custom 'P' pawn prefix."""
    text = re.sub(r'\{.*?\}', '', str(movetext))
    text = re.sub(r'\d+\.+\s*', '', text)
    raw_moves = text.split()
    formatted_moves = [apply_pawn_prefix(m) for m in raw_moves]
    return formatted_moves

def get_game_evaluations(clean_moves, engine, k, depth):
    """Runs Stockfish on every position in the game and returns a ';' separated string."""
    board = chess.Board()
    ply_predictions = []
    
    for move in clean_moves:
        analysis = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
        if isinstance(analysis, dict): analysis = [analysis]
        
        topk_san = [board.san(info.get("pv")[0]) for info in sorted(analysis, key=lambda x: x.get("multipv", 1)) if info.get("pv")]
        topk_formatted = [apply_pawn_prefix(m) for m in topk_san]
        ply_predictions.append(" ".join(topk_formatted))
        
        try:
            board.push_san(strip_pawn_prefix(move))
        except Exception:
            break
            
    return ";".join(ply_predictions)

# ==========================================
# 2. SCHEMA CONVERSION
# ==========================================
def transform_row(row, engine, args):
    """Maps to Kaggle schema AND adds optional Stockfish predictions."""
    clean_moves_list = clean_and_format_moves(row.get('movetext', ''))
    moves_str = " ".join(clean_moves_list)
    
    if engine:
        sf_preds = get_game_evaluations(clean_moves_list, engine, args.k, args.depth)
    else:
        sf_preds = ""

    try:
        dt_str = f"{row['UTCDate']} {row['UTCTime']}"
        timestamp = int(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    except:
        timestamp = 0

    res = str(row.get('Result', ''))
    winner = 'white' if res == '1-0' else 'black' if res == '0-1' else 'draw'
    term = str(row.get('Termination', '')).lower()
    
    if 'time' in term: status = 'outoftime'
    elif '#' in moves_str: status = 'mate'
    elif res == '1/2-1/2': status = 'draw'
    else: status = 'resign'

    return {
        "id": str(row.get('Site', '')).split('/')[-1],
        "rated": "Rated" in str(row.get('Event', '')),
        "created_at": timestamp,
        "last_move_at": timestamp,
        "turns": len(clean_moves_list),
        "victory_status": status,
        "winner": winner,
        "increment_code": row.get('TimeControl', ''),
        "white_id": row.get('White', ''),
        "white_rating": row.get('WhiteElo', 0),
        "black_id": row.get('Black', ''),
        "black_rating": row.get('BlackElo', 0),
        "moves": moves_str,
        "stockfish_predictions": sf_preds,
        "opening_eco": row.get('ECO', ''),
        "opening_name": row.get('Opening', ''),
        "opening_ply": 0
    }

def save_splits(df_dict, out_dir, fmt):
    for name, df in df_dict.items():
        if fmt in ["csv", "both"]:
            df.to_csv(out_dir / f"{name}.csv", index=False)
        if fmt in ["jsonl", "both"]:
            df.to_json(out_dir / f"{name}.jsonl", orient="records", lines=True)

# ==========================================
# 3. PER-FILE PROCESSING LOGIC
# ==========================================
def process_and_split_file(file_path, base_out_dir, engine, args, use_subfolder=True):
    """Processes a single CSV and saves its splits to a dedicated folder."""
    if use_subfolder:
        out_dir = base_out_dir / file_path.stem
    else:
        out_dir = base_out_dir
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Processing file: {file_path.name}")
    df_raw = pd.read_csv(file_path)
    
    if args.limit:
        df_raw = df_raw.head(args.limit)

    converted_data = []
    # Progress bar specifically for this file
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=f"Converting {file_path.name}"):
        converted_data.append(transform_row(row, engine, args))

    df_converted = pd.DataFrame(converted_data)
    df_converted = df_converted[df_converted['moves'].str.strip() != '']
    total_rows = len(df_converted)

    if total_rows == 0:
        print(f"[!] No valid chess games found in {file_path.name}. Skipping.")
        return

    # Split Data
    train_df, temp_df = train_test_split(df_converted, test_size=0.2, random_state=args.seed)
    eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    # Save
    df_dict = {"train": train_df, "eval": eval_df, "test": test_df}
    save_splits(df_dict, out_dir, args.output_format)

    print(f"  -> Saved splits to {out_dir.absolute()}/")
    print(f"  -> Train: {len(train_df):>6} | Eval: {len(eval_df):>6} | Test: {len(test_df):>6}")

# ==========================================
# 4. MAIN ROUTINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Convert to Kaggle schema, evaluate with Stockfish, and Split.")
    parser.add_argument("--input", required=True, help="Path to a raw CSV file OR a folder containing CSVs.")
    parser.add_argument("--output_dir", default="converted_splits", help="Base folder to save splits.")
    parser.add_argument("--output_format", choices=["csv", "jsonl", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--stockfish_path", default=None, help="Path to Stockfish binary (optional).")
    parser.add_argument("--depth", type=int, default=1, help="Engine search depth.")
    parser.add_argument("--k", type=int, default=5, help="Number of top moves to predict.")
    parser.add_argument("--limit", type=int, default=None, help="Max games to process per file (for testing).")

    args = parser.parse_args()
    
    input_path = Path(args.input)
    out_base_path = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"[!] Cannot find input path: {input_path}")

    # Initialize Engine Once (re-used across all files)
    engine = None
    if args.stockfish_path:
        if Path(args.stockfish_path).exists():
            print(f"[*] Starting Stockfish engine (Depth: {args.depth}, Top-K: {args.k})...")
            engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        else:
            print(f"[!] Warning: Stockfish not found at {args.stockfish_path}. Skipping evaluations.")
    else:
        print("[*] No Stockfish path provided. Skipping evaluations.")

    # Route logic based on file vs directory
    if input_path.is_file():
        if input_path.suffix.lower() != '.csv':
            print(f"[!] Warning: Provided file {input_path.name} does not end in .csv")
        # For a single file, just output directly to the requested output_dir
        process_and_split_file(input_path, out_base_path, engine, args, use_subfolder=False)
        
    elif input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"[!] No .csv files found in directory: {input_path}")
        
        print(f"[*] Found {len(csv_files)} CSV file(s) in directory. Processing individually...")
        for file_path in csv_files:
            # For a directory, create a subfolder for each CSV file's splits
            process_and_split_file(file_path, out_base_path, engine, args, use_subfolder=True)

    if engine:
        engine.quit()
        
    print("\n[*] All operations complete.")

if __name__ == "__main__":
    main()