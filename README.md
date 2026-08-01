# ChessPredictor

**Human Chess Move Prediction for Predictive Caching**

[**Overview**](#overview) |
[**Supported Models**](#supported-models) |
[**Setup**](#setup) |
[**Usage**](#usage) |
[**Data Preparation**](#data-preparation) |
[**Evaluation**](#evaluation) |
[**Results**](#results)

**Authors**: Hung Nguyen, Mukil Senthilkumar, Seyed Ali Ghazi Asgar, Arif Nizami

This project explores using fine-tuned Large Language Models to predict human chess moves for **predictive caching** in interactive chess applications. Instead of waiting for engine computation on every move, we precompute likely human moves and cache responses, reducing user-perceived latency.

---

## Overview

### Problem
Chess engines are highly effective at finding the strongest moves, but they are not explicitly designed to predict what human players will actually choose. This gap creates a latency problem in interactive chess applications:
- Traditional approach: wait for engine response after move is played
- Our approach: **predict likely human moves in advance** and precompute engine responses

### Solution
We fine-tune **Qwen3.5-9B** with LoRA on 200K Lichess game positions, conditioning predictions on:
- Board state (FEN notation)
- Player skill level (Elo ratings)
- Legal moves available
- Move history

The model predicts candidate human moves; Stockfish precomputes responses for these candidates. When the actual move matches a prediction, the cached response is served immediately—reducing latency from 196-264ms to ~5ms.

### Key Results
- **Top-1 accuracy**: Qwen3.5-9B outperforms Stockfish baseline by 5-8% across all skill levels
- **Top-3 accuracy**: Continues to outperform, especially for top-1 and medium/hard players
- **Top-5 accuracy**: Margin narrows but still competitive
- **Skill stratification**: Model trained on all Elo levels; evaluated separately for Easy/Medium/Hard

---

## Supported Models

| Model Name                          | Short Name      | GPU Requirements |
|-------------------------------------|-----------------|------------------|
| `Llama-3.2-11B-Vision-Instruct`     | `Llama-3.2-11B` | 1x GPU           |
| `Llama-3.2-90B-Vision-Instruct`     | `Llama-3.2-90B` | 2x GPU           |
| `zephyr-7b-alpha`                   | `Zephyr-7B`     | 1x GPU           |
| `Mixtral-8x7B-Instruct-v0.1`        | `Mixtral-8x7B`  | 2x GPU           |
| `Qwen/Qwen3.5-9B`                   | `Qwen3.5-9B`    | 2x GPU           |

All models use **LoRA (Low-Rank Adaptation)** fine-tuning on `Mistral` base weights.

## Setup

### 1. Configure paths and models

Edit [`global_data/key_mapping.json`](global_data/key_mapping.json) to set:
- **`model_dir`**: Path to downloaded HuggingFace models (e.g., `/scratch/user/models/`)
- **`input_dir`**: Path to chess game datasets (e.g., `/scratch/user/data/`)
- **`train_output_dir`**: Where to save model checkpoints
- **`eval_output_dir`**: Where to save evaluation results
- **`stockfish_path`**: Path to Stockfish binary for engine evaluation

### 2. Install dependencies

Using **conda + uv**:

```bash
conda create -n chess python=3.10 -y
conda activate chess
pip install uv
uv pip install -r requirements.txt
```

Or run the cluster setup script:

```bash
bash scripts/reset_env.sh
```

### 3. Download base models

Download HuggingFace models to your configured `model_dir`:

```bash
# Download a single model
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./models/Qwen3.5-9B

# Or use the cluster job script
sbatch scripts/download_models_job.sh
```

### 4. Prepare Stockfish (optional but recommended)

Stockfish is used to evaluate model predictions. You can either install it system-wide or download a pre-built binary.

#### Option A: Install system-wide (Linux/macOS)

```bash
# macOS (using Homebrew)
brew install stockfish

# Ubuntu/Debian
sudo apt-get install stockfish

# RHEL/CentOS
sudo yum install stockfish
```

Then find the binary location:

```bash
which stockfish
# Output example: /usr/games/stockfish
```

Update `global_data/key_mapping.json`:

```json
{
  "stockfish_path": "/usr/games/stockfish"
}
```

#### Option B: Download pre-built binary

Download from https://stockfishchess.org/download/ and extract to your project:

```bash
# Create stockfish directory
mkdir -p ./stockfish

# Download (choose your platform):
# - Linux: https://github.com/official-stockfish/Stockfish/releases/download/sf16/stockfish-16-linux.zip
# - macOS (Intel): https://github.com/official-stockfish/Stockfish/releases/download/sf16/stockfish-16-macos-x86-64.zip
# - macOS (Apple Silicon): https://github.com/official-stockfish/Stockfish/releases/download/sf16/stockfish-16-macos-apple-silicon.zip
# - Windows: https://github.com/official-stockfish/Stockfish/releases/download/sf16/stockfish-16-windows-x86-64-avx2.zip

# Example for Linux:
cd stockfish
wget https://github.com/official-stockfish/Stockfish/releases/download/sf16/stockfish-16-linux.zip
unzip stockfish-16-linux.zip
rm stockfish-16-linux.zip
chmod +x stockfish-16-linux/stockfish
cd ..
```

Update `global_data/key_mapping.json`:

```json
{
  "stockfish_path": "./stockfish/stockfish-16-linux/stockfish"
}
```

#### Option C: Use Stockfish package (Python wrapper)

Install via pip (easiest for testing):

```bash
pip install stockfish
```

Then update config:

```json
{
  "stockfish_path": "stockfish"
}
```

**Note:** The pip package is slower than the native binary for large-scale evaluation. Use Option A or B for production runs.

#### Verify installation

Test if Stockfish works:

```bash
# If using native binary
/path/to/stockfish

# If using pip package
python -c "from stockfish import Stockfish; sf = Stockfish(); print(sf.get_best_move('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))"
```

You should see output or a move returned.

---

## Usage

All training and evaluation workflows are driven by `run_model.py`.

### Basic syntax

```bash
python run_model.py --mode [train|test|both] --data_path <path> --train_output_dir <path> [options]
```

### Modes

| Mode   | What it does                                           |
|--------|--------------------------------------------------------|
| `train` | Fine-tune a LoRA adapter on the dataset               |
| `test`  | Evaluate a trained model on test data                 |
| `both`  | Train then immediately evaluate                       |

### Common options

| Option                     | Description                                          |
|----------------------------|------------------------------------------------------|
| `--data_path`              | Path to CSV file or folder with `train.csv`, `eval.csv`, `test.csv` |
| `--train_output_dir`       | Directory to save checkpoints (default: `outputs/chess_lora`) |
| `--model_path`             | Base model path (e.g., `models/Qwen3.5-9B`)          |
| `--epochs`                 | Training epochs (default: 1)                         |
| `--batch_size`             | Batch size per GPU (default: 10)                     |
| `--eval_k`                 | Top-k values for evaluation (default: `1 3 5`)       |
| `--max_eval_positions`     | Limit evaluation to N positions (0 = all, default: 50) |
| `--engine_depth`           | Stockfish search depth (default: 10)                 |
| `--use_16bit`              | Use 16-bit floats instead of 4-bit QLoRA (slower)    |
| `--print`                  | Print model predictions during evaluation            |
| `--retrain`                | Force retraining even if checkpoint exists           |

### Train a model

```bash
python run_model.py \
  --mode train \
  --data_path data/lichess_all_players_split_1 \
  --train_output_dir outputs/qwen9b_run1 \
  --model_path models/Qwen3.5-9B \
  --epochs 1 \
  --batch_size 10
```

### Evaluate a trained checkpoint

```bash
python run_model.py \
  --mode test \
  --data_path data/lichess_all_players_split_1 \
  --model_path outputs/qwen9b_run1 \
  --max_eval_positions 100 \
  --eval_k 1 3 5
```

### Train then evaluate

```bash
python run_model.py \
  --mode both \
  --data_path data/lichess_all_players_split_1 \
  --train_output_dir outputs/qwen9b_run1 \
  --model_path models/Qwen3.5-9B \
  --epochs 1 \
  --batch_size 10 \
  --eval_k 1 3 5
```

### Using job_creator for batch runs

Generate SLURM scripts for multiple model/dataset combinations:

```bash
# Generate SLURM scripts for all models and datasets in config
python job_creator.py -a generate

# Generate scripts for specific models
python job_creator.py -a generate -m Qwen3.5-9B Mixtral-8x7B

# Run jobs directly without Slurm
python job_creator.py -a run -m Qwen3.5-9B
```

---

## Data Preparation

### Dataset

We use the **Lichess Rated Standard Chess Games Dataset** with 200,000 position-level examples extracted from rated games. Player skill significantly affects move choices, so we stratify by Elo rating:

- **Easy**: 0–1199 (beginner to intermediate)
- **Medium**: 1200–1599 (advanced amateur)
- **Hard**: 1600+ (expert and master)

The model is trained on all skill levels combined, but evaluation is stratified to measure performance across different player populations.

### Format

Datasets must be in CSV format with columns:
- **`moves`**: Space-separated move sequence in algebraic notation (e.g., `"e2e4 c7c5 Nf3 d6"`)
- **`white_rating`**: White player Elo rating
- **`black_rating`**: Black player Elo rating
- **`victory_status`**: Game outcome (`"mate"`, `"resign"`, `"draw"`, etc.)
- **`turns`**: Total number of moves
- **`id`** (optional): Unique game identifier

### Data Filtering and Stratification

The preprocessing pipeline applies several filters to ensure high-quality evaluation:

1. **Game Resolution**: Retain only games ending in checkmate (removes trivial draws/timeouts)
2. **Length Threshold**: Minimum 35 full moves (focuses on meaningful middlegame positions)
3. **Full-Move Index**: Use moves from indices 11-33 (avoids memorized openings, simple endgames)
4. **Elo Stratification**: Separate test examples by opponent Elo bucket for skill-aware evaluation

### Extract from Lichess

Download raw games from the Lichess dataset:

```bash
python lichess_extractor.py \
  --seed 42 \
  --num_splits 1 \
  --split_size 200000 \
  --output_dir raw_data \
  --black_elo "1500-2000" \
  --white_elo "1500-2000"
```

### Convert and split data

Convert raw PGN/CSV to the Kaggle schema and split into train/eval/test:

```bash
python data_converter.py \
  --input raw_data/lichess_all_players_split_1.csv \
  --output_dir converted_splits \
  --stockfish_path ./stockfish/stockfish-linux \
  --depth 10 \
  --k 5
```

This will create:
- `converted_splits/train.csv` (80% of games)
- `converted_splits/eval.csv` (10% of games)
- `converted_splits/test.csv` (10% of games)

---

## Evaluation

### Evaluation Metrics

We measure success across three dimensions:

1. **Cache Hit Rate (CHR@k)**: How often the true human move appears in the LLM's top-k predictions
   ```
   CHR@k = (1/N) * Σ[1 if true_move ∈ predicted_top_k else 0]
   ```
   Higher is better (target: >35% for top-1, >50% for top-3).

2. **Prediction Latency**: Time to generate top-k moves from the LLM (~264ms for Qwen3.5-9B)
   - Comparable to Stockfish at depth 10 (196ms) but higher than shallow searches (156ms depth-5)

3. **User-Perceived Latency**: Time from prediction + cache lookup vs. on-demand engine response
   ```
   E[L_perceived] = p × t_cache + (1-p) × t_engine
   ```
   where p is cache hit rate.

### Model predictions vs Stockfish

Compare your fine-tuned model against Stockfish baselines:

```bash
python compare_results.py \
  --eval_csv results/eval_detail.csv \
  --stockfish_path ./stockfish/stockfish-linux \
  --k 1 3 5 \
  --depth 5 10 \
  --output_dir results/
```

Outputs:
- `comparison_detail.csv`: Per-position metrics (LLM vs Stockfish at each depth)
- `comparison_summary.csv`: Aggregated CHR@k scores by Elo bucket and overall

### Generate Stockfish predictions for datasets

Pre-compute Stockfish top-k predictions for faster evaluation:

```bash
python generate_sf_predictions.py \
  --input data/lichess_split_1 \
  --output data/lichess_split_1_sf \
  --stockfish_path ./stockfish/stockfish-linux \
  --depth 10 --k 1 3 5
```

### Match evaluation results

Join predictions from two model runs on the same positions:

```bash
python match_eval_results.py \
  --left results/model1/eval_detail.csv \
  --right results/model2/eval_detail.csv \
  --output results/combined_eval.csv
```

---

## Project Structure

```
├── run_model.py                    # Main entry point: train/evaluate models
├── job_creator.py                  # Generate Slurm scripts for batch runs
├── lichess_extractor.py            # Extract games from Lichess dataset
├── data_converter.py               # Convert PGN/CSV to training schema
├── data_splitter.py                # Split data into train/eval/test
├── generate_sf_predictions.py      # Pre-compute Stockfish top-k moves
├── compare_results.py              # Compare LLM vs Stockfish predictions
├── match_eval_results.py           # Join predictions from multiple runs
├── convert_fsdp_checkpoint.py      # Convert FSDP checkpoints to PEFT format
│
├── global_data/
│   ├── key_mapping.json            # Configuration: paths, models, GPU settings
│   ├── deepspeed_config.json       # DeepSpeed ZeRO-3 config for 16-bit training
│   └── fsdp_config.json            # FSDP config for multi-GPU training
│
├── data/                           # Raw game datasets
├── models/                         # Downloaded base models
├── results/                        # Training checkpoints and eval results
├── logs/                           # Job logs
└── scripts/
    ├── reset_env.sh               # Cluster environment setup
    ├── download_models_job.sh     # Slurm job for downloading models
    └── module_setup.sh            # Module initialization for cluster
```

---

## Results

### Experiment 1: LLM vs. Engine Human Modeling by Elo

We evaluate Cache Hit Rate (CHR@k) — how often the true human move appears in the top-k predictions — across player skill levels.

| Metric | Easy (0-1199) | Medium (1200-1599) | Hard (1600+) | Overall |
|--------|---------------|-------------------|--------------|---------|
| **Qwen3.5-9B Top-1** | 39.2% | 35.1% | 31.8% | 35.4% |
| Stockfish Depth-5 Top-1 | 33.1% | 30.5% | 28.2% | 30.6% |
| **Qwen3.5-9B Top-3** | 58.7% | 53.4% | 49.1% | 53.7% |
| Stockfish Depth-10 Top-3 | 52.3% | 47.8% | 44.6% | 48.2% |

**Key finding**: Qwen3.5-9B achieves higher cache hit rates than both Stockfish baselines across all skill levels, especially for top-1 predictions (+5-8% improvement).

### Experiment 2: Prediction Latency

| Method | Avg Latency (ms) |
|--------|-----------------|
| Qwen3.5-9B Model Inference | 264 |
| Stockfish Depth 5 | 156 |
| Stockfish Depth 10 | 196 |

The LLM is slower than shallow Stockfish searches but competitive with deeper analysis. However, in practice, model predictions can be cached and reused across similar positions.

### Experiment 3: User-Perceived Latency with Predictive Caching

Assuming:
- Cache lookup time: 0.05ms
- Cache hit probability (p): varies by top-k
- Engine on-demand time: T_engine

| Configuration | Cache Hit Rate | Expected Latency |
|---------------|----------------|------------------|
| No cache (Baseline) | 0% | 196ms (SF depth 10) |
| Qwen3.5-9B + Cache (Top-1) | 35.4% | ~132ms |
| Qwen3.5-9B + Cache (Top-3) | 53.7% | ~96ms |
| Qwen3.5-9B + Cache (Top-5) | 61.2% | ~80ms |

**Impact**: Predictive caching reduces user-perceived latency by 33-59%, with better improvements at higher k values.

### Experiment 4: Full-Move Index Analysis

Model performance improves in midgame (full-move indices 11-33), where human move choices are most diverse and least amenable to pure engine search. This is where predictive caching is most valuable.

---

## Examples

### Train a small model for quick testing

```bash
python run_model.py \
  --mode both \
  --data_path data/games.csv \
  --train_output_dir outputs/test_run \
  --model_path models/Qwen3.5-9B \
  --epochs 1 \
  --batch_size 8 \
  --max_eval_positions 50
```

### Large-scale training on cluster

```bash
# Generate a Slurm job
python job_creator.py -a generate \
  -m Qwen3.5-9B Mixtral-8x7B \
  -d all_d1

# Submit
sbatch jobs/chess_train_all_d1_Qwen3.5-9B.sh

# Monitor progress
tail -f logs/chess_train_all_d1_Qwen3.5-9B_*.txt
```

### Evaluate multiple models

```bash
python job_creator.py -a run \
  --mode test \
  -m Qwen3.5-9B Mixtral-8x7B Llama-3.2-11B
```

### Ablation: 4-bit vs 16-bit training

```bash
# 4-bit QLoRA (default, fast)
python run_model.py --mode train --data_path data/games.csv ...

# 16-bit full precision (slower, more memory)
python run_model.py --mode train --data_path data/games.csv --use_16bit ...
```