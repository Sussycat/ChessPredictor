"""
Chess Move Predictor — train and evaluate a LoRA-finetuned Mistral model.

Usage
-----
# Train from a single CSV
python run_model.py train --data_path data/games.csv --train_output_dir outputs/run1

# Train from a pre-split folder (train.csv / eval.csv / test.csv)
python run_model.py train --data_path data/lichess_all_players_split_1_s42_d1 --train_output_dir outputs/run1

# Evaluate a saved checkpoint
python run_model.py test --data_path data/games.csv --model_path outputs/run1/checkpoint-238

# Train then immediately evaluate
python run_model.py both --data_path data/games.csv --train_output_dir outputs/run1
"""

import argparse
import logging
import os
import random
import re
import subprocess
import time
from pathlib import Path

import chess
import chess.engine
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stockfish helpers
# ---------------------------------------------------------------------------

def detect_stockfish_path():
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    search_roots = ["/usr", "/usr/games", "/usr/local/bin", "/usr/bin", "/bin"]
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        try:
            result = subprocess.run(
                ["find", str(root_path), "-name", "stockfish", "-type", "f"],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        for line in result.stdout.splitlines():
            found = line.strip()
            if found:
                return found
    return None


# ---------------------------------------------------------------------------
# Chess / move helpers
# ---------------------------------------------------------------------------

def add_p_to_pawn_moves_str(moves_str: str) -> str:
    result = []
    for m in moves_str.split():
        if m in {"O-O", "O-O-O"}:
            result.append(m)
        elif m[0] not in {"N", "B", "Q", "R", "K", "O"}:
            result.append("P" + m if not m.startswith("P") else m)
        else:
            result.append(m)
    return " ".join(result)


def add_p_to_pawn_moves_list(moves):
    result = []
    for m in moves:
        if m in {"O-O", "O-O-O"}:
            result.append(m)
        elif m and m[0] not in {"N", "B", "Q", "R", "K", "O"}:
            result.append("P" + m if not m.startswith("P") else m)
        else:
            result.append(m)
    return result


def get_legal_moves(moves_str, san=True):
    board = chess.Board()
    for move_str in moves_str.split():
        move_str = move_str.strip()
        if move_str.startswith("P") and move_str not in {"O-O", "O-O-O"}:
            move_str = move_str[1:]
        try:
            board.push_san(move_str)
        except Exception:
            return []
    return [board.san(m) for m in board.legal_moves] if san else [m.uci() for m in board.legal_moves]


def get_stockfish_topk_moves(moves_str, engine, k=10, depth=1):
    board = chess.Board()
    for move_str in moves_str.split():
        if move_str.startswith("P") and move_str not in {"O-O", "O-O-O"}:
            move_str = move_str[1:]
        try:
            board.push_san(move_str)
        except Exception:
            return []
    analysis = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    if isinstance(analysis, dict):
        analysis = [analysis]
    moves = []
    for info in sorted(analysis, key=lambda x: x.get("multipv", 1)):
        pv = info.get("pv", [])
        if pv:
            moves.append(board.san(pv[0]))
    return moves


def extract_piece_and_target_square(san_move):
    if not san_move:
        return None
    san = re.sub(r"[+#?!]", "", str(san_move).strip())
    san = san.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    if san in {"O-O", "O-O-O"}:
        return san
    san_no_promo = san.split("=")[0]
    m = re.search(r"([a-h][1-8])(?!.*[a-h][1-8])", san_no_promo)
    if not m:
        return san
    piece = san_no_promo[0] if san_no_promo[0] in "KQRBN" else "P"
    return f"{piece}{m.group(1)}"


def normalize_label(text):
    if not text:
        return None
    text = str(text).strip().replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    text = text.splitlines()[0].strip().rstrip(".,;:")
    if " " in text:
        text = text.split()[0]
    if re.fullmatch(r"[pnbrqk][a-h][1-8]", text):
        text = text[0].upper() + text[1:]
    if text in {"O-O", "O-O-O"} or re.fullmatch(r"[PNBRQK][a-h][1-8]", text):
        return text
    return extract_piece_and_target_square(text)


def unique_in_order(values):
    seen, result = set(), []
    for v in values:
        if v is not None and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def get_board_after_moves(moves_played):
    board = chess.Board()
    for move_str in str(moves_played).split():
        move_str = move_str.strip()
        if not move_str:
            continue
        if move_str.startswith("P") and move_str not in {"O-O", "O-O-O"}:
            move_str = move_str[1:]
        move_str = re.sub(r"[?!]", "", move_str)
        board.push_san(move_str)
    return board


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_examples(df_mate, seed=42):
    black_moves = list(range(11, 35, 2))  # odd indices = black's moves
    examples = []
    for row in df_mate.dropna(subset=["moves"]).itertuples(index=False):
        tokens = str(row.moves).strip().split()
        valid = [m for m in black_moves if m < len(tokens)]
        if not valid:
            continue

        sf_col = getattr(row, "stockfish_predictions", None)
        ply_preds = sf_col.split(";") if sf_col and isinstance(sf_col, str) and sf_col.strip() else []
        valid_with_sf = [m for m in valid if m < len(ply_preds) and ply_preds[m].strip()]
        if ply_preds and not valid_with_sf:
            continue  # predictions exist but all cut off before any black-move position

        # Use every valid black-move position as a separate training/eval example
        positions = valid_with_sf if valid_with_sf else valid
        for prompt_end in positions:
            precomputed_sf = ply_preds[prompt_end].strip() if prompt_end < len(ply_preds) else ""
            examples.append({
                "game_id": getattr(row, "game_id", None),
                "prompt": " ".join(tokens[:prompt_end]),
                "label": tokens[prompt_end],
                "turn_index": prompt_end,
                "precomputed_sf": precomputed_sf,
                "white_rating": getattr(row, "white_rating", 0),
                "black_rating": getattr(row, "black_rating", 0),
            })
    return examples


def build_prompt(prompt_moves, engine=None, k=10, depth=1, precomputed_sf="", force_stockfish=False,
                 white_rating=0, black_rating=0):
    if precomputed_sf and not force_stockfish:
        sf_with_p = precomputed_sf.split()
    else:
        if engine is None:
            raise RuntimeError(
                "Stockfish engine required but not available. "
                "Pass --stockfish_path or ensure precomputed predictions exist in the dataset."
            )
        sf_moves = get_stockfish_topk_moves(prompt_moves, engine, k=k, depth=depth)
        sf_with_p = add_p_to_pawn_moves_list(sf_moves)
    clean_game = re.sub(r"[?!]", "", prompt_moves)
    prompt_text = (
        "Chess move prediction.\n"
        f"White Elo: {white_rating} | Black Elo: {black_rating}\n"
        f"Game: {clean_game}\n"
        f"Legal: {' '.join(sf_with_p)}\n"
        "Next move from legal list: "
    )
    return prompt_text, sf_with_p


def make_preprocess_fn(tokenizer, engine=None, max_length=280, force_stockfish=False):
    def preprocess(example):
        prompt_text, _ = build_prompt(
            example["prompt"], engine,
            precomputed_sf=example.get("precomputed_sf", ""),
            force_stockfish=force_stockfish,
            white_rating=example.get("white_rating", 0),
            black_rating=example.get("black_rating", 0),
        )
        label_text = extract_piece_and_target_square(example["label"])
        if label_text is None:
            return None

        prompt_enc = tokenizer(prompt_text, add_special_tokens=True, padding=False, truncation=False)
        label_enc = tokenizer(label_text, add_special_tokens=False, padding=False, truncation=False)

        input_ids = (prompt_enc["input_ids"] + label_enc["input_ids"]
                     + [tokenizer.eos_token_id])[:max_length]
        unpadded_len = len(input_ids)
        pad_len = max_length - unpadded_len
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * unpadded_len + [0] * pad_len

        labels = ([-100] * len(prompt_enc["input_ids"])
                  + label_enc["input_ids"]
                  + [tokenizer.eos_token_id])[:max_length]
        labels = labels + [-100] * (max_length - len(labels))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return preprocess


def build_split_indices(frame, eval_fraction=0.1, seed=42):
    if "game_id" in frame.columns and frame["game_id"].notna().any():
        unique_games = frame["game_id"].astype(str).unique().tolist()
        rng = random.Random(seed)
        rng.shuffle(unique_games)
        n_eval = max(1, int(len(unique_games) * eval_fraction))
        eval_ids = set(unique_games[-n_eval:])
        eval_idx = frame.index[frame["game_id"].astype(str).isin(eval_ids)].tolist()
        train_idx = frame.index[~frame["game_id"].astype(str).isin(eval_ids)].tolist()
    else:
        indices = list(frame.index)
        rng = random.Random(seed)
        rng.shuffle(indices)
        n_eval = max(1, int(len(indices) * eval_fraction))
        eval_idx = sorted(indices[-n_eval:])
        train_idx = sorted(indices[:-n_eval])
    return train_idx, eval_idx


def _load_and_filter_csv(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["moves"]).copy()
    df["moves"] = df["moves"].apply(add_p_to_pawn_moves_str)
    return df[(df["victory_status"] == "mate") & (df["turns"] > 35)]


def get_data_splits(data_path, seed=42):
    """Return (train_ex_df, eval_ex_df, test_ex_df) as DataFrames of examples.

    Folder: reads train.csv, eval.csv (or valid.csv), and test.csv directly —
            no further splitting is performed.
    File:   reads the CSV, filters, builds examples, then splits 90/10
            (eval and test share the same slice for single-file inputs).
    """
    path = Path(data_path)
    if path.is_dir():
        eval_csv = path / "eval.csv" if (path / "eval.csv").exists() else path / "valid.csv"
        train_ex = pd.DataFrame(build_examples(_load_and_filter_csv(path / "train.csv"), seed=seed))
        eval_ex  = pd.DataFrame(build_examples(_load_and_filter_csv(eval_csv), seed=seed))
        test_ex  = pd.DataFrame(build_examples(_load_and_filter_csv(path / "test.csv"), seed=seed))
        log.info("Folder splits — train=%d | eval=%d | test=%d",
                 len(train_ex), len(eval_ex), len(test_ex))
    else:
        df_filtered = _load_and_filter_csv(path)
        log.info("Mate games (>35 turns): %d", len(df_filtered))
        df_sup = pd.DataFrame(build_examples(df_filtered, seed=seed))
        train_idx, eval_idx = build_split_indices(df_sup, eval_fraction=0.1, seed=seed)
        train_ex = df_sup.iloc[train_idx].reset_index(drop=True)
        eval_ex  = df_sup.iloc[eval_idx].reset_index(drop=True)
        test_ex  = eval_ex
        log.info("File splits — train=%d | eval/test=%d", len(train_ex), len(eval_ex))
    return train_ex, eval_ex, test_ex


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def save_peft_from_fsdp(fsdp_checkpoint_dir, base_model_path, output_dir):
    """Merge FSDP shards and save LoRA adapter in PEFT format (rank 0 only)."""
    import re
    import json
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    from safetensors.torch import save_file

    fsdp_dir = os.path.join(fsdp_checkpoint_dir, "pytorch_model_fsdp_0")
    if not os.path.isdir(fsdp_dir):
        log.warning("FSDP shard dir not found, skipping PEFT save: %s", fsdp_dir)
        return

    merged_path = os.path.join(output_dir, "_merged_tmp.pt")
    log.info("Merging FSDP shards from %s ...", fsdp_dir)
    dcp_to_torch_save(fsdp_dir, merged_path)
    state_dict = torch.load(merged_path, map_location="cpu", weights_only=False)
    os.remove(merged_path)

    adapter_state = {}
    for key, tensor in state_dict.items():
        if "lora_" not in key:
            continue
        clean = re.sub(r"^_fsdp_wrapped_module\.", "", key)
        clean = clean.replace("._fsdp_wrapped_module.", ".")
        adapter_state[clean] = tensor.contiguous()

    if not adapter_state:
        log.warning("No LoRA keys found in FSDP checkpoint — adapter not saved.")
        return

    save_file(adapter_state, os.path.join(output_dir, "adapter_model.safetensors"))

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_path,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    log.info("PEFT adapter saved to %s", output_dir)


def load_base_model(model_id, use_4bit=True):
    # bnb_4bit_quant_storage=bfloat16 lets FSDP shard the 4-bit layers as bfloat16 tensors.
    # device_map={"": local_rank} loads each rank's full copy to its own GPU so FSDP can shard correctly.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    local_rank = _local_rank()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb if use_4bit else None,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_lora_checkpoint(checkpoint_path, use_4bit=True):
    from peft import PeftConfig
    base_model_id = PeftConfig.from_pretrained(checkpoint_path).base_model_name_or_path
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    local_rank = _local_rank()
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb if use_4bit else None,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def generate_model_topk(prompt_moves, model, tokenizer, k=5, max_new_tokens=8,
                        precomputed_sf="", white_rating=0, black_rating=0):
    prompt_text, _ = build_prompt(
        prompt_moves, engine=None,
        precomputed_sf=precomputed_sf, force_stockfish=False,
        white_rating=white_rating, black_rating=black_rating,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(k, 2),
            num_return_sequences=min(k, max(k, 2)),
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    preds = []
    for output in outputs:
        text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
        norm = normalize_label(text)
        if norm and norm not in preds:
            preds.append(norm)
    return preds


def evaluate_llm(eval_examples_df, model, tokenizer,
                 k_values=(1, 3, 5), max_positions=None):
    ordered_k = tuple(sorted(set(k_values)))
    max_k = max(ordered_k)
    subset = eval_examples_df.reset_index(drop=True)
    if max_positions:
        subset = subset.head(max_positions)

    rows = []
    for i, (_, row) in enumerate(subset.iterrows()):
        actual = normalize_label(row["label"])
        precomputed_sf = row.get("precomputed_sf", "") if hasattr(row, "get") else ""

        t0 = time.time()
        llm_preds = generate_model_topk(
            row["prompt"], model, tokenizer, k=max_k,
            precomputed_sf=precomputed_sf,
            white_rating=row.get("white_rating", 0),
            black_rating=row.get("black_rating", 0),
        )
        llm_ms = (time.time() - t0) * 1000

        rec = {
            "game_id": row.get("game_id"),
            "prompt": row["prompt"],
            "actual_label": actual,
            "precomputed_sf": precomputed_sf,
            "llm_inference_ms": llm_ms,
        }
        for k in ordered_k:
            rec[f"llm_hit@{k}"] = actual in llm_preds[:k]
            rec[f"llm_top_{k}"] = ", ".join(llm_preds[:k])
        rows.append(rec)

        if (i + 1) % 10 == 0:
            log.info("  %d/%d positions evaluated", i + 1, len(subset))

    detail_df = pd.DataFrame(rows)
    summary_rows = []
    for k in ordered_k:
        summary_rows.append({
            "k": k,
            "positions": len(detail_df),
            "llm_pass@k": detail_df[f"llm_hit@{k}"].mean(),
            "llm_avg_ms": detail_df["llm_inference_ms"].mean(),
        })
    return detail_df, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Train / test entry points
# ---------------------------------------------------------------------------

def run_train(args):
    log.info("Loading dataset: %s", args.data_path)
    train_ex_df, eval_ex_df, test_ex_df = get_data_splits(args.data_path, seed=args.seed)

    model, tokenizer = load_base_model(args.model_path, use_4bit=not args.use_16bit)

    # Use precomputed Stockfish predictions from the dataset when available.
    # The engine is still needed for eval baseline; for preprocessing it is
    # only required when --force_stockfish is set or any example lacks predictions.
    all_precomputed = all(
        "precomputed_sf" in df.columns
        and df["precomputed_sf"].str.strip().str.len().gt(0).all()
        for df in [train_ex_df, eval_ex_df]
    )
    need_engine_for_preprocess = args.force_stockfish or not all_precomputed

    sf_path = args.stockfish_path or detect_stockfish_path()
    if sf_path is None and need_engine_for_preprocess:
        raise RuntimeError("Stockfish not found. Install it or pass --stockfish_path.")
    if sf_path is None:
        log.info("No Stockfish found; using precomputed predictions for preprocessing.")
    else:
        log.info("Stockfish: %s", sf_path)

    train_raw = Dataset.from_pandas(train_ex_df.reset_index(drop=True))
    eval_raw  = Dataset.from_pandas(eval_ex_df.reset_index(drop=True))

    def _run_map(engine_):
        force = args.force_stockfish if engine_ is not None else False
        fn = make_preprocess_fn(tokenizer, engine_, max_length=args.max_output_length, force_stockfish=force)
        tr = train_raw.map(fn, remove_columns=train_raw.column_names)
        ev = eval_raw.map(fn, remove_columns=eval_raw.column_names)
        return tr, ev

    if need_engine_for_preprocess:
        with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
            train_dataset, eval_dataset = _run_map(engine)
    else:
        train_dataset, eval_dataset = _run_map(None)
    log.info("Train=%d | Eval=%d", len(train_dataset), len(eval_dataset))

    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import f1_score
        pred_ids, label_ids = eval_pred
        pred_ids = np.array(pred_ids)
        label_ids = np.array(label_ids)
        # Fallback: if preprocess_logits_for_metrics didn't run, pred_ids is [batch, seq, vocab]
        if pred_ids.ndim == 3:
            pred_ids = pred_ids.argmax(axis=-1)
        all_preds, all_labels = [], []
        skipped = 0
        for pred, label in zip(pred_ids, label_ids):
            non_masked = np.where(label != -100)[0]
            if len(non_masked) == 0 or non_masked[0] == 0:
                skipped += 1
                continue
            pos = int(non_masked[0])
            all_preds.append(int(pred[pos - 1]))
            all_labels.append(int(label[pos]))
        log.info("compute_metrics: used=%d skipped=%d", len(all_labels), skipped)
        if not all_labels:
            return {"accuracy": 0.0, "f1": 0.0}
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return {"accuracy": accuracy, "f1": f1}

    Path(args.train_output_dir).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.train_output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=args.epochs,
            bf16=True,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=10,
            gradient_checkpointing=True,
            report_to="none",
            optim="adamw_torch",
            deepspeed=args.deepspeed_config if hasattr(args, "deepspeed_config") else None,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    log.info("Training done. Outputs: %s", args.train_output_dir)

    if _local_rank() == 0:
        best_ckpt = trainer.state.best_model_checkpoint or args.train_output_dir
        peft_out = os.path.join(args.train_output_dir, "final_model")
        save_peft_from_fsdp(best_ckpt, args.model_path, peft_out)

    return model, tokenizer, test_ex_df, sf_path


def run_test(args, model=None, tokenizer=None, eval_examples_df=None, sf_path=None):
    if _local_rank() != 0:
        return

    if model is None:
        if not args.model_path:
            raise ValueError("--model_path is required for test mode.")
        final_model_path = os.path.join(args.model_path, "final_model")
        if os.path.isdir(final_model_path):
            log.info("Found final_model folder, using: %s", final_model_path)
            args.model_path = final_model_path
        log.info("Loading checkpoint: %s", args.model_path)
        model, tokenizer = load_lora_checkpoint(args.model_path, use_4bit=not args.use_16bit)

    if eval_examples_df is None:
        log.info("Rebuilding test set from %s", args.data_path)
        _, _, eval_examples_df = get_data_splits(args.data_path, seed=args.seed)
        eval_examples_df = eval_examples_df.reset_index(drop=True)

    n_eval = min(args.max_eval_positions, len(eval_examples_df)) if args.max_eval_positions else len(eval_examples_df)
    log.info("Evaluating %d positions...", n_eval)

    detail_df, summary_df = evaluate_llm(
        eval_examples_df=eval_examples_df,
        model=model,
        tokenizer=tokenizer,
        k_values=tuple(args.eval_k),
        max_positions=args.max_eval_positions or None,
    )

    log.info("\n%s\nLLM PASS@K SUMMARY\n%s\n%s",
             "=" * 60, summary_df.round(4).to_string(index=False), "=" * 60)

    eval_out = args.eval_output_dir or args.train_output_dir
    if eval_out:
        Path(eval_out).mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(Path(eval_out) / "eval_detail.csv", index=False)
        summary_df.to_csv(Path(eval_out) / "eval_summary.csv", index=False)
        log.info("Eval CSVs saved to %s", eval_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Chess move predictor: train and/or evaluate a LoRA-finetuned Mistral model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["train", "test", "both"], help="Operation mode.")

    # Required paths
    p.add_argument("--data_path", required=True,
                   help="Path to a games CSV file, or a folder containing "
                        "train.csv, eval.csv (or valid.csv), and test.csv.")
    p.add_argument("--model_path", default=None,
                   help="Checkpoint dir to load for test/both. Defaults to <output_dir>/checkpoint-last.")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Path to a checkpoint to resume training from. "
                        "Pass 'true' to resume from the latest checkpoint in --train_output_dir.")
    p.add_argument("--train_output_dir", default="outputs/chess_lora",
                   help="Directory for saved model checkpoints and training logs.")
    p.add_argument("--eval_output_dir", default=None,
                   help="Directory for eval CSVs. Defaults to --train_output_dir if not set.")

    # Model config
    p.add_argument("--stockfish_path", default=None,
                   help="Path to Stockfish binary (auto-detected if omitted).")

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_output_length", type=int, default=280,
                   help="Max tokenized sequence length.")

    # Evaluation
    p.add_argument("--eval_k", type=int, nargs="+", default=[1, 3, 5],
                   help="k values for top-k evaluation.")
    p.add_argument("--max_eval_positions", type=int, default=50,
                   help="Max positions to evaluate (0 = all).")
    p.add_argument("--engine_depth", type=int, default=10,
                   help="Stockfish search depth used during evaluation.")

    p.add_argument("--eval_steps", type=float, default=0.1,
                   help="Eval/save frequency: fraction of epoch (0.0–1.0) or integer steps.")
    # Quantization
    p.add_argument("--use_16bit", action="store_true",
                   help="Load model in 16-bit (float16). Default is 4-bit NF4 (QLoRA).")
    p.add_argument("--deepspeed_config", default=None,
                   help="Path to DeepSpeed ZeRO-3 config JSON for 16-bit multi-GPU training.")

    # Stockfish override
    p.add_argument("--force_stockfish", action="store_true",
                   help="Always run the Stockfish engine even when precomputed predictions "
                        "exist in the dataset column 'stockfish_predictions'.")

    # Auth
    p.add_argument("--hf_token", default=None,
                   help="HuggingFace token (falls back to HF_TOKEN / HUGGINGFACE_TOKEN env vars).")

    return p.parse_args()


def main():
    args = parse_args()

    if args.max_eval_positions == 0:
        args.max_eval_positions = None

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        log.info("Logged into HuggingFace.")

    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:  # both
        model, tokenizer, test_ex_df, sf_path = run_train(args)
        run_test(args, model=model, tokenizer=tokenizer,
                 eval_examples_df=test_ex_df, sf_path=sf_path)


if __name__ == "__main__":
    main()
