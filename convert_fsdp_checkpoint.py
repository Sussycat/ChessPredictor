"""
Convert an FSDP sharded checkpoint to a PEFT adapter checkpoint.

The FSDP Trainer saves weights as .distcp shards (pytorch_model_fsdp_0/).
This script reassembles them and extracts the LoRA adapter weights,
saving adapter_config.json + adapter_model.safetensors so the checkpoint
can be loaded with PeftModel.from_pretrained().

Usage (run on the cluster, single GPU is fine):
    python convert_fsdp_checkpoint.py \
        --checkpoint_dir /scratch/user/nguye3hv/checkpoints/all_d1/Qwen3.5-9B/checkpoint-1612 \
        --base_model /scratch/user/nguye3hv/models/Qwen3.5-9B \
        --output_dir /scratch/user/nguye3hv/checkpoints/all_d1/Qwen3.5-9B/checkpoint-1612-peft
"""

import argparse
import json
import os
import re
import threading

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from safetensors.torch import save_file
from tqdm import tqdm


LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "peft_type": "LORA",
    "base_model_name_or_path": None,  # filled in at runtime
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to checkpoint-XXXX directory containing pytorch_model_fsdp_0/.")
    parser.add_argument("--base_model", required=True,
                        help="Base model path (used in adapter_config.json).")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save PEFT adapter files. Defaults to <parent of checkpoint_dir>/final_model.")
    args = parser.parse_args()

    fsdp_dir = os.path.join(args.checkpoint_dir, "pytorch_model_fsdp_0")
    if not os.path.isdir(fsdp_dir):
        raise FileNotFoundError(f"FSDP shard directory not found: {fsdp_dir}")

    out_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint_dir), "final_model")
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: merge FSDP shards into a single state dict file
    merged_path = os.path.join(out_dir, "_merged_state_dict.pt")
    done = threading.Event()

    def _spinner():
        with tqdm(desc="Merging FSDP shards", bar_format="{desc} {elapsed}", leave=False) as pbar:
            while not done.wait(0.5):
                pbar.update()

    t = threading.Thread(target=_spinner, daemon=True)
    t.start()
    dcp_to_torch_save(fsdp_dir, merged_path)
    done.set()
    t.join()

    state_dict = torch.load(merged_path, map_location="cpu", weights_only=False)
    os.remove(merged_path)
    print(f"Merged {len(state_dict)} keys.")

    # Step 2: extract LoRA adapter weights
    lora_keys = sorted(k for k in state_dict if "lora_" in k)
    print(f"Found {len(lora_keys)} LoRA parameter tensors.")

    adapter_state = {}
    for key in tqdm(lora_keys, desc="Extracting LoRA weights"):
        clean = re.sub(r"^_fsdp_wrapped_module\.", "", key)
        clean = clean.replace("._fsdp_wrapped_module.", ".")
        adapter_state[clean] = state_dict[key].contiguous()

    if not adapter_state:
        raise RuntimeError("No LoRA keys found in the checkpoint. "
                           "The checkpoint may not contain trained adapter weights.")

    # Step 3: save adapter_model.safetensors
    safetensors_path = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(adapter_state, safetensors_path)
    print(f"Saved adapter weights -> {safetensors_path}")

    # Step 4: save adapter_config.json
    config = dict(LORA_CONFIG)
    config["base_model_name_or_path"] = args.base_model
    config_path = os.path.join(out_dir, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved adapter config  -> {config_path}")

    print("\nDone. Load the converted checkpoint with:")
    print(f"  PeftModel.from_pretrained(base_model, '{out_dir}')")


if __name__ == "__main__":
    main()
