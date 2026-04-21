import json
import sys
import os
import itertools
import argparse
import subprocess

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

main_dir = os.path.dirname(os.path.realpath(__file__))

config_path = os.path.join(main_dir, "global_data", "key_mapping.json")
try:
    with open(config_path) as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find config at {config_path}")
    sys.exit(1)

TARGET_SCRIPT = os.path.join(main_dir, "run_model.py")

SLURM_TEMPLATE = """#!/bin/bash

# --- SLURM RESOURCE REQUESTS ---
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:{num_gpus}
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}

# --- SETUP ENVIRONMENT ---
module load CUDA/11.8.0
source /sw/eb/sw/Miniconda3/23.10.0-1/bin/activate {conda_env}

cd {work_dir}

# --- RUN SCRIPT ---
echo "Starting run on $(hostname) with GPUs: $CUDA_VISIBLE_DEVICES"
echo "Job: {job_name}"
echo "Command: {cmd_str}"

{cmd_str}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smart_path(path):
    if not path:
        return ""
    if path.startswith("/"):
        return path
    return os.path.join(main_dir, path)


def get_config_details(model_key, dataset_key, mode):
    # Model path
    base_model_dir = smart_path(CONFIG.get("model_dir", ""))
    model_subpath = CONFIG.get("model_alts", {}).get(model_key, model_key)
    model_path = os.path.join(base_model_dir, model_subpath)

    # GPU settings
    gpu_map = CONFIG.get("gpu_mappings", {}).get("default", {})
    gpu_list = gpu_map.get(model_key, gpu_map.get("default", [0]))
    num_gpus = len(gpu_list)
    gpu_comma_str = ",".join(str(g) for g in gpu_list)

    # Dataset path (folder with pre-split train/eval/test CSVs)
    base_data_dir = smart_path(CONFIG.get("input_dir", "data/"))
    dataset_subpath = CONFIG.get("dataset_alts", {}).get(dataset_key, dataset_key)
    dataset_path = os.path.join(base_data_dir, dataset_subpath)

    # Output path for this run
    base_output_dir = smart_path(CONFIG.get("output_dir", "results/"))
    safe_model = model_key.replace("/", "_")
    output_path = os.path.join(base_output_dir, mode, dataset_key, safe_model)

    # LoRA checkpoint path: always the train output for this dataset/model combo
    checkpoint_path = os.path.join(base_output_dir, "train", dataset_key, safe_model)

    stockfish_path = smart_path(CONFIG.get("stockfish_path", ""))
    max_eval_positions = CONFIG.get("max_eval_positions", None)
    engine_depth = CONFIG.get("engine_depth", None)

    return {
        "model_path": model_path,
        "checkpoint_path": checkpoint_path,
        "stockfish_path": stockfish_path,
        "max_eval_positions": max_eval_positions,
        "engine_depth": engine_depth,
        "num_gpus": num_gpus,
        "gpu_comma_str": gpu_comma_str,
        "dataset_path": dataset_path,
        "output_path": output_path,
        "epochs": CONFIG.get("epochs", 1),
        "batch_size": CONFIG.get("batch_size", 8),
        "seed": CONFIG.get("seed", 42),
        "eval_k": CONFIG.get("eval_k", [1, 3, 5]),
    }


def validate_compatibility(dataset_key, model_key):
    """Returns False if the config task_mapping explicitly disallows this combination."""
    for task, task_datasets in CONFIG.get("task_mapping", {}).get("dataset", {}).items():
        if dataset_key not in task_datasets:
            continue
        allowed_models = CONFIG.get("task_mapping", {}).get("model", {}).get(task)
        if allowed_models is not None and model_key not in allowed_models:
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts or run jobs for the chess move predictor."
    )
    parser.add_argument("-a", "--action", choices=["generate", "run"], default="generate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode", choices=["train", "test", "both"], default="both",
        help="run_model.py mode written into every generated script.",
    )
    parser.add_argument("-d", "--dataset", nargs="+", help="Dataset keys to include (default: all).")
    parser.add_argument("-m", "--model", nargs="+", help="Model keys to include (default: all).")
    parser.add_argument("--conda_env", default="chess",
                        help="Conda environment name to activate in the SLURM script.")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token passed to run_model.py.")
    parser.add_argument("--slurm-dir", default=os.path.join(main_dir, "jobs"))
    args = parser.parse_args()

    if args.dry_run:
        print("\n--- DRY RUN MODE ---\n")

    all_datasets = list(CONFIG.get("dataset_alts", {}).keys())
    all_models = list(CONFIG.get("model_alts", {}).keys())

    datasets = args.dataset if args.dataset else all_datasets
    models = args.model if args.model else all_models

    slurm_dir = args.slurm_dir
    log_dir = os.path.join(main_dir, "logs")

    if args.action == "generate" and not args.dry_run:
        os.makedirs(slurm_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Scripts will be saved to: {slurm_dir}")

    combinations = list(itertools.product(datasets, models))
    print(f"Processing {len(combinations)} combinations | action={args.action} | mode={args.mode}")

    count = 0
    for dataset_key, model_key in combinations:
        if not validate_compatibility(dataset_key, model_key):
            print(f"  [SKIP] {dataset_key} x {model_key} — not in task_mapping")
            continue

        cfg = get_config_details(model_key, dataset_key, args.mode)
        safe_model = model_key.replace("/", "_").replace(" ", "")
        job_name = f"chess_{args.mode}_{dataset_key}_{safe_model}"

        # Build command
        cmd_list = [
            "python", TARGET_SCRIPT,
            args.mode,
            "--data_path",  cfg["dataset_path"],
            "--output_dir", cfg["output_path"],
            "--model_id",   cfg["model_path"],
            "--epochs",     str(cfg["epochs"]),
            "--batch_size", str(cfg["batch_size"]),
            "--seed",       str(cfg["seed"]),
            "--eval_k",
        ] + [str(k) for k in cfg["eval_k"]]

        if args.mode in ("test", "both"):
            cmd_list += ["--model_path", cfg["checkpoint_path"]]
        if cfg["stockfish_path"]:
            cmd_list += ["--stockfish_path", cfg["stockfish_path"]]
        if args.hf_token:
            cmd_list += ["--hf_token", args.hf_token]
        if cfg["max_eval_positions"] is not None:
            cmd_list += ["--max_eval_positions", str(cfg["max_eval_positions"])]
        if cfg["engine_depth"] is not None:
            cmd_list += ["--engine_depth", str(cfg["engine_depth"])]

        cmd_str = " ".join(str(c) for c in cmd_list)
        log_filename = os.path.join(log_dir, f"{job_name}_%j.txt")

        if args.action == "generate":
            slurm_filename = os.path.join(slurm_dir, f"{job_name}.sh")
            script_content = SLURM_TEMPLATE.format(
                num_gpus=cfg["num_gpus"],
                job_name=job_name,
                log_path=log_filename,
                work_dir=main_dir,
                cmd_str=cmd_str,
                conda_env=args.conda_env,
            )
            if args.dry_run:
                print(f"\n[DRY] Preview: {slurm_filename}")
                print(script_content.strip())
            else:
                with open(slurm_filename, "w") as f:
                    f.write(script_content)
                print(f"[GEN] {slurm_filename}")

        elif args.action == "run":
            if args.dry_run:
                print(f"[DRY] Would execute: {cmd_str}")
            else:
                print(f"[RUN] {job_name}")
                my_env = os.environ.copy()
                my_env["CUDA_VISIBLE_DEVICES"] = cfg["gpu_comma_str"]
                try:
                    subprocess.run(cmd_list, env=my_env, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running {job_name}: {e}")
                except KeyboardInterrupt:
                    print("\nStopped by user.")
                    sys.exit(1)

        count += 1

    if args.dry_run:
        print(f"\nDry run complete. {count} valid combinations.")
    elif args.action == "generate":
        print(f"\nGenerated {count} SLURM scripts in {slurm_dir}/")
        print(f"To submit all:  for f in {slurm_dir}/*.sh; do sbatch \"$f\"; done")
    else:
        print(f"\nExecuted {count} jobs.")


if __name__ == "__main__":
    main()
