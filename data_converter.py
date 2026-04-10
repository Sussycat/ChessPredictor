import argparse
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

def clean_moves(movetext):
    """Removes clock metadata and move numbers."""
    text = re.sub(r'\{.*?\}', '', str(movetext))
    text = re.sub(r'\d+\.+\s*', '', text)
    moves = text.split()
    return " ".join(moves), len(moves)

def transform_row(row):
    """Maps a single row from HF schema to Kaggle schema."""
    moves_str, turn_count = clean_moves(row.get('movetext', ''))
    
    # Timestamp logic
    try:
        dt_str = f"{row['UTCDate']} {row['UTCTime']}"
        timestamp = int(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    except:
        timestamp = 0

    # Victory logic
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
        "turns": turn_count,
        "victory_status": status,
        "winner": winner,
        "increment_code": row.get('TimeControl', ''),
        "white_id": row.get('White', ''),
        "white_rating": row.get('WhiteElo', 0),
        "black_id": row.get('Black', ''),
        "black_rating": row.get('BlackElo', 0),
        "moves": moves_str,
        "opening_eco": row.get('ECO', ''),
        "opening_name": row.get('Opening', ''),
        "opening_ply": 0
    }

def process_folder(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return

    print(f"Found {len(csv_files)} files. Starting conversion...")

    for file in csv_files:
        print(f"Processing {file.name}...")
        
        # Load the raw CSV
        df_raw = pd.read_csv(file)
        
        # Apply transformation to every row
        converted_data = [transform_row(row) for _, row in df_raw.iterrows()]
        
        # Save to new folder
        output_file = output_path / f"converted_{file.name}"
        pd.DataFrame(converted_data).to_csv(output_file, index=False)
        print(f"  Done -> {output_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert Lichess CSVs to Kaggle format.")
    parser.add_argument("--input_dir", default="raw_data", help="Folder containing raw CSVs")
    parser.add_argument("--output_dir", default="converted_data", help="Folder for converted CSVs")
    
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)
    print("\nAll conversions complete.")