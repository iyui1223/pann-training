"""
Extract hidden-layer activations from trained PANN models (k-fold).

For each fold, runs the fold's model on its validation set, then
concatenates across folds so every sample has activations from the
model that did NOT train on it.

Usage:
    python extract_hidden.py --model_dir Data/M04_train_model/models \
                             --data Data/M03_dataset/training_data_partitioned/training_dataset_partitioned.nc \
                             --out_dir Data/M04_train_model/hidden_activations
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import load_raw_arrays, kfold_indices, make_dataloaders
from evaluate import _build_model


def extract_hidden_activations(model, loader, device):
    """Forward pass collecting hidden activations from all branches."""
    model.eval()
    names = model.branch_names
    accum = {name: [] for name in names}
    branch_outputs = {name: [] for name in names}

    with torch.no_grad():
        for batch in loader:
            *xb, _y = batch
            xb = [t.to(device) for t in xb]
            _, branches = model(*xb, save_hidden=True)
            for name in names:
                out, hiddens = branches[name]
                branch_outputs[name].append(out.cpu().numpy())
                accum[name].append([h.numpy() for h in hiddens])

    result = {}
    for name in names:
        n_layers = len(accum[name][0])
        result[name] = {
            "hidden": [
                np.concatenate([batch[i] for batch in accum[name]])
                for i in range(n_layers)
            ],
            "output": np.concatenate(branch_outputs[name]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract PANN hidden activations (k-fold)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(model_dir / "config_used.yaml") as f:
        cfg = yaml.safe_load(f)

    n_folds = cfg.get("n_folds", 5)

    raw = load_raw_arrays(
        args.data,
        n_levels=cfg.get("n_levels"),
        level_start=cfg.get("level_start"),
        target_cfg=cfg.get("target"),
    )
    folds = kfold_indices(raw["n_samples"], n_folds, cfg.get("random_seed", 42))
    branch_names = list(raw["branch_names"])

    all_val_idx = []
    combined = {
        name: {"hidden_layers": None, "output": []}
        for name in branch_names
    }

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        fold_dir = model_dir / f"fold_{fold_i}"
        print(f"Fold {fold_i}: {len(val_idx)} validation samples")

        ckpt = torch.load(fold_dir / "best_model.pt", map_location=device,
                           weights_only=False)
        fold_cfg = ckpt.get("config", cfg)

        data = make_dataloaders(raw, train_idx, val_idx, cfg["batch_size"])

        model = _build_model(fold_cfg, raw["n_levels"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        acts = extract_hidden_activations(model, data["val_loader"], device)
        all_val_idx.append(val_idx)

        for name in branch_names:
            combined[name]["output"].append(acts[name]["output"])
            if combined[name]["hidden_layers"] is None:
                combined[name]["hidden_layers"] = [
                    [h] for h in acts[name]["hidden"]
                ]
            else:
                for layer_list, h in zip(combined[name]["hidden_layers"],
                                          acts[name]["hidden"]):
                    layer_list.append(h)

    val_idx_all = np.concatenate(all_val_idx)

    for branch_name, bd in combined.items():
        save_dict = {"output": np.concatenate(bd["output"])}
        for i, layer_chunks in enumerate(bd["hidden_layers"]):
            save_dict[f"hidden_layer_{i}"] = np.concatenate(layer_chunks)
        out_path = out_dir / f"hidden_{branch_name}.npz"
        np.savez_compressed(out_path, **save_dict)
        shapes_str = ", ".join(f"{k}: {v.shape}" for k, v in save_dict.items())
        print(f"Saved {out_path}  ({shapes_str})")

    np.savez(out_dir / "sample_order.npz", val_idx=val_idx_all)
    print(f"Sample order saved to {out_dir / 'sample_order.npz'}")


if __name__ == "__main__":
    main()
