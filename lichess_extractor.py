import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

def parse_elo(val):
    """Parses ELO strings into a [min, max] list."""
    if not val:
        return None
    val = val.strip()

    if val.endswith('+'):
        return [int(val[:-1]), 4000]
    if val.endswith('-'):
        return [0, int(val[:-1])]
    if '-' in val:
        parts = val.split('-')
        return [int(parts[0].strip()), int(parts[1].strip())]
    
    return [int(val), int(val)]

def main():
    parser = argparse.ArgumentParser(description="Extract randomized, filtered subsets from Lichess with HF Auth.")
    
    # Configuration
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling.")
    parser.add_argument("--num_splits", type=int, default=1, help="Number of files to create.")
    parser.add_argument("--split_size", type=int, default=200000, help="Games per file.")
    parser.add_argument("--output_dir", type=str, default="raw_data", help="Output directory.")
    parser.add_argument("--hf_token_path", type=str, default=None, help="Path to a .txt file containing your HF token.")
    
    # ELO Filters
    parser.add_argument("--black_elo", type=str, default=None, 
                        help="Black ELO filter: '200+', '200-', or '1500-2000'.")
    parser.add_argument("--white_elo", type=str, default=None, 
                        help="White ELO filter: '200+', '200-', or '1500-2000'.")

    args = parser.parse_args()

    # --- HUGGING FACE LOGIN ---
    if args.hf_token_path:
        token_path = Path(args.hf_token_path)
        if token_path.exists():
            token = token_path.read_text().strip()
            login(token=token)
            print(f"Logged in successfully using token from: {args.hf_token_path}")
        else:
            print(f"Warning: Token file not found at {args.hf_token_path}. Proceeding without login.")

    # --- PRINT ARGUMENTS ---
    print("="*40)
    print("LICHESS EXTRACTION PARAMETERS")
    print("-" * 40)
    for arg, value in vars(args).items():
        # Mask the token path for slight privacy in logs if you want, or just print it
        print(f"{arg.replace('_', ' ').title():<15}: {value}")
    print("="*40)

    b_range = parse_elo(args.black_elo)
    w_range = parse_elo(args.white_elo)

    # 1. Directory Setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Dataset Stream
    print(f"\nConnecting to Lichess... (Target: {out_dir.absolute()})")
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)

    # 3. Apply Filtering
    filename_tags = []

    if b_range:
        print(f"Filter active: Black ELO {b_range[0]} to {b_range[1]}")
        ds = ds.filter(lambda x: x['BlackElo'] is not None and b_range[0] <= x['BlackElo'] <= b_range[1])
        filename_tags.append(f"black_{args.black_elo.replace('+', 'plus').replace('-', 'minus')}")

    if w_range:
        print(f"Filter active: White ELO {w_range[0]} to {w_range[1]}")
        ds = ds.filter(lambda x: x['WhiteElo'] is not None and w_range[0] <= x['WhiteElo'] <= w_range[1])
        filename_tags.append(f"white_{args.white_elo.replace('+', 'plus').replace('-', 'minus')}")

    tag = "_".join(filename_tags) if filename_tags else "all_players"

    # 4. Shuffle & Iterate
    shuffled_ds = ds.shuffle(seed=args.seed, buffer_size=10000)
    iterator = iter(shuffled_ds)

    for i in range(1, args.num_splits + 1):
        print(f"\n--- Generating Split {i}/{args.num_splits} ---")
        chunk = []
        
        while len(chunk) < args.split_size:
            try:
                chunk.append(next(iterator))
                if len(chunk) % 25000 == 0:
                    print(f"  Progress: {len(chunk)} / {args.split_size} games fetched...")
            except StopIteration:
                print("Reached the end of the dataset.")
                break
        
        if not chunk:
            break
            
        # 5. Save to CSV
        df = pd.DataFrame(chunk)
        filename = f"lichess_{tag}_split_{i}_s{args.seed}.csv"
        df.to_csv(out_dir / filename, index=False)
        print(f"Successfully saved: {filename}")

    print("\nMission Complete. All subsets extracted.")

if __name__ == "__main__":
    main()