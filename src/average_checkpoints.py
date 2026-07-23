"""
Average weights across multiple checkpoint files to produce an ensembled checkpoint.
Averaging 3 to 5 top checkpoints consistently yields +0.5 to +1.5 BLEU score improvements in NMT.

Usage:
  python src/average_checkpoints.py --checkpoints checkpoints/epoch_10.pt checkpoints/epoch_11.pt checkpoints/epoch_12.pt --output checkpoints/best_model_averaged.pt
"""

import torch
import argparse
import os
import sys

def average_checkpoints(checkpoint_paths, output_path):
    print("=" * 60)
    print("Checkpoint Averaging Engine")
    print("=" * 60)
    print(f"Number of checkpoints to average: {len(checkpoint_paths)}")
    for path in checkpoint_paths:
        print(f"  - {path}")
    print()

    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided.")

    avg_state_dict = None
    num_models = len(checkpoint_paths)
    base_checkpoint = None

    for i, path in enumerate(checkpoint_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        if i == 0:
            base_checkpoint = ckpt
            avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k, v in state_dict.items():
                if k in avg_state_dict:
                    avg_state_dict[k] += v.float()
                else:
                    print(f"Warning: Key {k} missing from base checkpoint.")

    for k in avg_state_dict:
        avg_state_dict[k] /= num_models

    output_checkpoint = {
        "model_state_dict": avg_state_dict,
        "averaged_from": checkpoint_paths,
        "num_checkpoints_averaged": num_models
    }

    if isinstance(base_checkpoint, dict):
        if "epoch" in base_checkpoint:
            output_checkpoint["epoch"] = base_checkpoint["epoch"]
        if "val_loss" in base_checkpoint:
            output_checkpoint["val_loss"] = base_checkpoint["val_loss"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_checkpoint, output_path)
    print(f"Successfully saved averaged checkpoint to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average weights across PyTorch model checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint file paths to average")
    parser.add_argument("--output", default="checkpoints/best_model_averaged.pt", help="Path to save the averaged checkpoint")

    args = parser.parse_args()
    average_checkpoints(args.checkpoints, args.output)
