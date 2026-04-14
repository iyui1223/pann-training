"""
PANN training script with k-fold cross-validation.

Supports level-blocking: when ``--block_idx`` is given, only the specified
block of levels is trained.  Models are saved under ``block_B/fold_F/``.

Usage (from shell script or directly):
    python train.py --config Const/M04_train_model/config.yaml \
                    --data   Data/M03_dataset/training_data_partitioned/training_dataset_partitioned.nc \
                    --save_dir Data/M04_train_model/models \
                    [--block_idx 0]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import (
    load_raw_arrays,
    compute_blocks,
    kfold_indices,
    make_dataloaders,
    save_branch_scalers,
)
from model import PANN


def get_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def make_level_weights(pressure_pa, n_output_vars):
    """Weight per level proportional to p/p_surface, normalised so mean = 1.

    Equivalent to exp(-z/H) density weighting for an isothermal atmosphere.
    """
    p = np.asarray(pressure_pa, dtype=np.float64).ravel()
    w = p / p.max()
    w = w / w.mean()
    w = np.tile(w, n_output_vars)
    return torch.as_tensor(w, dtype=torch.float32)


def _eval_val_loss(model, val_loader, loss_fn, device):
    """Compute mean validation loss (no gradient)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            *xb, y = batch
            xb = [t.to(device) for t in xb]
            y = y.to(device)
            pred = model(*xb)
            total += loss_fn(pred, y).item() * y.size(0)
    return total / len(val_loader.dataset)


def _branch_variance_reweight(model, val_loader, device):
    """Reweight branches by explained variance of their predictions.

    For each branch, compute the mean variance of its output across
    samples (averaged over output dimensions).  A branch that predicts
    near-constant values -- regardless of magnitude -- gets a small
    importance score, while a branch whose output varies meaningfully
    across samples is considered important.

    Weights are normalised so they sum to N (number of branches),
    preserving the overall output magnitude on average.
    """
    model.eval()
    branch_outs = {n: [] for n in model.branch_names}

    with torch.no_grad():
        for batch in val_loader:
            *xb, y = batch
            xb = [t.to(device) for t in xb]
            _, branches = model(*xb, save_hidden=True)
            for name in model.branch_names:
                branch_outs[name].append(branches[name][0].cpu().numpy())

    branch_var = {}
    for name in model.branch_names:
        arr = np.concatenate(branch_outs[name])
        branch_var[name] = float(arr.var(axis=0).mean())

    total_var = sum(branch_var.values())
    n_branches = len(model.branch_names)
    weights = {
        n: branch_var[n] / max(total_var, 1e-30) * n_branches
        for n in model.branch_names
    }

    log = {"branches": {}}
    for name in model.branch_names:
        log["branches"][name] = {
            "variance": branch_var[name],
            "fraction": branch_var[name] / max(total_var, 1e-30),
            "weight": weights[name],
        }

    return weights, log


def train_one_fold(model, train_loader, val_loader, cfg, device, save_dir,
                   level_weights=None):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    if level_weights is not None:
        w = level_weights.to(device)
        def loss_fn(pred, target):
            return (w * (pred - target) ** 2).mean()
    else:
        _mse = torch.nn.MSELoss()
        def loss_fn(pred, target):
            return _mse(pred, target)

    # Shapley branch reweighting config
    bd_cfg = cfg.get("branch_dropout") or {}
    warmup_epochs = int(bd_cfg.get("warmup_epochs", 0))
    reweight_interval = int(bd_cfg.get("reweight_interval", 5))
    reweight_log = None

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            *xb, y = batch
            xb = [t.to(device) for t in xb]
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(*xb)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * y.size(0)

        avg_train = total_train_loss / len(train_loader.dataset)

        # Variance-based branch reweighting: periodic after warmup
        past_warmup = (epoch + 1) >= warmup_epochs and warmup_epochs > 0
        at_interval = ((epoch + 1 - warmup_epochs) % reweight_interval == 0
                       if past_warmup else False)
        if past_warmup and at_interval:
            print(f"\n  --- Variance branch reweight after epoch {epoch + 1} ---")
            weights, reweight_log = _branch_variance_reweight(
                model, val_loader, device,
            )
            for bname, info in reweight_log["branches"].items():
                print(f"    {bname:16s}  var={info['variance']:.6e}  "
                      f"frac={info['fraction']:.4f}  weight={info['weight']:.4f}")
            model.set_branch_weights(weights)
            print()

        avg_val = _eval_val_loss(model, val_loader, loss_fn, device)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(
            f"  Epoch {epoch + 1:4d}/{cfg['epochs']}  "
            f"train_loss={avg_train:.6e}  val_loss={avg_val:.6e}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
                "config": cfg,
                "branch_mask": model.branch_mask.copy(),
                "branch_weights": model.branch_weights.copy(),
            }
            if reweight_log is not None:
                checkpoint["reweight_log"] = reweight_log
            torch.save(checkpoint, save_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    return history


def main():
    parser = argparse.ArgumentParser(description="Train PANN (k-fold)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--block_idx", type=int, default=None,
                        help="Train only this level-block (0-based index)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(args.device or cfg.get("device", "cpu"))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    save_dir = Path(args.save_dir)

    block_idx = args.block_idx
    block_size = cfg.get("block_size")
    block_overlap = cfg.get("block_overlap")

    if block_idx is not None:
        save_dir = save_dir / f"block_{block_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_folds = cfg.get("n_folds", 5)

    raw = load_raw_arrays(
        args.data,
        n_levels=cfg.get("n_levels"),
        level_start=cfg.get("level_start"),
        target_cfg=cfg.get("target"),
        block_idx=block_idx,
        block_size=block_size,
        block_overlap=block_overlap,
    )
    folds = kfold_indices(raw["n_samples"], n_folds, cfg.get("random_seed", 42))

    names = raw["branch_names"]
    n_output_vars = int(raw.get("n_output_vars", 2))
    cfg["n_output_vars"] = n_output_vars
    target_mode = raw.get("target_mode", "scm_minus_reference")

    if block_idx is not None:
        cfg["block_idx"] = block_idx
        cfg["block_start"] = raw.get("block_start")
        cfg["block_end"] = raw.get("block_end")
        cfg["full_n_levels"] = raw.get("full_n_levels")
        print(f"Block {block_idx}: levels [{cfg['block_start']}, {cfg['block_end']})")

    print(
        f"Samples: {raw['n_samples']}, Levels: {raw['n_levels']}, Folds: {n_folds}, "
        f"target_mode={target_mode}  n_output_vars={n_output_vars}"
    )
    print(f"Partitioned branches ({len(names)}): {names}")
    for i, n in enumerate(names):
        dim = raw["x_branch_list"][i].shape[1]
        nvar = raw["n_input_vars"][n]
        print(f"  {n:16s}  input_dim={dim:5d}  (~{nvar} vars/level)")

    cfg["branch_names"] = list(names)
    cfg["branch_input_dims"] = {
        n: int(raw["x_branch_list"][i].shape[1])
        for i, n in enumerate(names)
    }

    # Level weighting for loss
    level_weight_mode = cfg.get("level_weight", "uniform")
    level_weights = None
    if level_weight_mode == "pressure":
        pressure_pa = raw.get("pressure_pa")
        if pressure_pa is not None:
            level_weights = make_level_weights(pressure_pa, n_output_vars)
            print(f"Level weighting: pressure (p/p_sfc), "
                  f"range [{level_weights.min():.3f}, {level_weights.max():.3f}]")
        else:
            print("Warning: pressure weighting requested but no pressure_pa available; "
                  "falling back to uniform.")
    else:
        print("Level weighting: uniform")

    np.savez(
        save_dir / "fold_indices.npz",
        **{f"train_{i}": t for i, (t, _) in enumerate(folds)},
        **{f"val_{i}": v for i, (_, v) in enumerate(folds)},
    )
    with open(save_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    branches_spec = [
        (n, raw["x_branch_list"][i].shape[1]) for i, n in enumerate(names)
    ]

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_i}/{n_folds - 1}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        fold_dir = save_dir / f"fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        data = make_dataloaders(raw, train_idx, val_idx, cfg["batch_size"])

        model = PANN(
            n_levels=raw["n_levels"],
            hidden_dim=cfg["hidden_dim"],
            branches=branches_spec,
            architecture=cfg.get("architecture", "flat"),
            n_hidden_layers=cfg.get("n_hidden_layers", 4),
            bottleneck_dim=cfg.get("bottleneck_dim", 64),
            dropout=cfg.get("dropout", 0.0),
            n_output_vars=n_output_vars,
            conv_channels=cfg.get("conv_channels"),
        )
        if fold_i == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {n_params:,}")

        history = train_one_fold(
            model, data["train_loader"], data["val_loader"],
            cfg, device, fold_dir,
            level_weights=level_weights,
        )

        save_branch_scalers(
            data["scalers"], list(names), fold_dir / "scalers.npz",
        )
        np.savez(
            fold_dir / "training_history.npz",
            train_loss=np.array(history["train_loss"]),
            val_loss=np.array(history["val_loss"]),
        )

    elapsed = time.time() - t0
    print(f"\nAll {n_folds} folds finished in {elapsed / 60:.1f} min")
    print(f"Artefacts saved to {save_dir}")


if __name__ == "__main__":
    main()
