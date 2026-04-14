"""
Evaluate a trained PANN model (k-fold cross-validation).

Supports two layouts:
- **Standard**: models in ``model_dir/fold_F/``
- **Level-blocked**: models in ``model_dir/block_B/fold_F/``.
  Auto-detected by the presence of ``block_0/`` under ``model_dir``.
  Predictions from each block are stitched with linear blending in
  overlap zones to reconstruct the full vertical profile.

Usage:
    python evaluate.py --model_dir Data/M04_train_model/models \
                       --data Data/M03_dataset/training_data_partitioned/training_dataset_partitioned.nc \
                       --fig_dir Figs/M04_train_model
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml

from dataset import (
    load_raw_arrays,
    compute_blocks,
    _slice_raw_to_block,
    kfold_indices,
    make_dataloaders,
    load_branch_scalers,
    branch_label,
    TARGET_MODE_SIMPLE_LINEAR_Q,
    TARGET_MODE_SIMPLE_GAUSSIAN_Q,
    TARGET_MODE_CEDA_DQ,
)
from model import PANN


def _load_scalers(path, cfg):
    return load_branch_scalers(path, cfg["branch_names"])


def _predict(model, loader, device):
    """Return (predictions, targets) in normalised space."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            *xb, y = batch
            xb = [t.to(device) for t in xb]
            pred = model(*xb)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _predict_branches(model, loader, device):
    """Return per-branch outputs in normalised space, weighted by branch_weights.

    The model's forward pass stores raw (unweighted) branch outputs in
    ``branch_outputs``.  We multiply by the branch weight here so that
    downstream analysis (Shapley profiles, contribution plots) reflects
    each branch's actual contribution to the final prediction.
    """
    model.eval()
    branch_preds = {n: [] for n in model.branch_names}
    with torch.no_grad():
        for batch in loader:
            *xb, y = batch
            xb = [t.to(device) for t in xb]
            _, branches = model(*xb, save_hidden=True)
            for name in model.branch_names:
                w = model.branch_weights.get(name, 1.0)
                branch_preds[name].append(
                    (w * branches[name][0]).cpu().numpy())
    return {n: np.concatenate(v) for n, v in branch_preds.items()}


def _build_model(cfg, n_levels):
    if "branch_names" not in cfg or "branch_input_dims" not in cfg:
        raise ValueError(
            "config_used.yaml must list branch_names and branch_input_dims "
            "(retrain with current train.py)."
        )
    branches = [(n, cfg["branch_input_dims"][n]) for n in cfg["branch_names"]]
    return PANN(
        n_levels=n_levels,
        hidden_dim=cfg["hidden_dim"],
        branches=branches,
        architecture=cfg.get("architecture", "flat"),
        n_hidden_layers=cfg.get("n_hidden_layers", 4),
        bottleneck_dim=cfg.get("bottleneck_dim", 64),
        dropout=cfg.get("dropout", 0.0),
        n_output_vars=cfg.get("n_output_vars", 2),
        conv_channels=cfg.get("conv_channels"),
    )


def _load_model_from_checkpoint(ckpt, cfg, n_levels, device):
    """Build model, load weights, apply branch mask and weights if present."""
    fold_cfg = ckpt.get("config", cfg)
    model = _build_model(fold_cfg, n_levels)
    model.load_state_dict(ckpt["model_state_dict"])
    branch_mask = ckpt.get("branch_mask")
    if branch_mask:
        model.set_branch_mask(branch_mask)
        dropped = [n for n, v in branch_mask.items() if not v]
        if dropped:
            print(f"    Branch mask applied: dropped {dropped}")
    branch_weights = ckpt.get("branch_weights")
    if branch_weights:
        model.set_branch_weights(branch_weights)
        wstr = ", ".join(f"{n}={w:.4f}" for n, w in branch_weights.items())
        print(f"    Branch weights applied: {wstr}")
    model.to(device)
    return model, fold_cfg


# ── Metrics ─────────────────────────────────────────────────────────────────


def rmse_per_level(pred, target, n_levels, n_output_vars=2):
    pred_2d = pred.reshape(pred.shape[0], n_output_vars, n_levels)
    tgt_2d = target.reshape(target.shape[0], n_output_vars, n_levels)
    return np.sqrt(np.mean((pred_2d - tgt_2d) ** 2, axis=0))


def r2_per_level(pred, target, n_levels, n_output_vars=2):
    pred_2d = pred.reshape(pred.shape[0], n_output_vars, n_levels)
    tgt_2d = target.reshape(target.shape[0], n_output_vars, n_levels)
    ss_res = np.sum((tgt_2d - pred_2d) ** 2, axis=0)
    ss_tot = np.sum((tgt_2d - tgt_2d.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-20, 1.0 - ss_res / ss_tot, np.nan)
    return r2


# ── Y-axis helper ───────────────────────────────────────────────────────────


def _y_axis(pressure_pa, n_levels):
    """Return (y_values, ylabel) for profile plots.

    Uses pressure in hPa when available, otherwise model level index.
    """
    if pressure_pa is not None:
        return pressure_pa / 100.0, "Pressure (hPa)"
    return np.arange(1, n_levels + 1), "Model level"


# ── Block stitching ─────────────────────────────────────────────────────────


def _build_blend_weights(blocks, n_levels):
    """Build per-level blend weights for each block.

    Returns list of arrays, one per block, each of shape (block_levels,).
    In overlap zones weights ramp linearly; elsewhere weight = 1.
    """
    weights = []
    for b_idx, (bstart, bend) in enumerate(blocks):
        blen = bend - bstart
        w = np.ones(blen, dtype=np.float64)

        if b_idx > 0:
            prev_end = blocks[b_idx - 1][1]
            n_overlap = prev_end - bstart
            if n_overlap > 0:
                ramp = np.linspace(0, 1, n_overlap + 2)[1:-1]
                w[:n_overlap] = ramp

        if b_idx < len(blocks) - 1:
            next_start = blocks[b_idx + 1][0]
            n_overlap = bend - next_start
            if n_overlap > 0:
                ramp = np.linspace(1, 0, n_overlap + 2)[1:-1]
                w[-n_overlap:] = ramp

        weights.append(w)
    return weights


def _stitch_blocks(block_preds, blocks, blend_weights, n_levels, n_out):
    """Stitch per-block predictions into full (n_samples, n_out * n_levels)."""
    n_samples = block_preds[0].shape[0]
    accum = np.zeros((n_samples, n_out, n_levels), dtype=np.float64)
    wsum = np.zeros((n_out, n_levels), dtype=np.float64)

    for b_idx, (bstart, bend) in enumerate(blocks):
        w = blend_weights[b_idx]
        pred = block_preds[b_idx]
        pred_3d = pred.reshape(n_samples, n_out, bend - bstart)
        for v in range(n_out):
            accum[:, v, bstart:bend] += pred_3d[:, v, :] * w[None, :]
            wsum[v, bstart:bend] += w

    wsum[wsum == 0] = 1.0
    accum /= wsum[None, :, :]
    return accum.reshape(n_samples, n_out * n_levels)


def _evaluate_one_block(block_dir, raw_full, cfg_main, device, block_idx,
                        block_size, block_overlap):
    """Predict for one block across all folds, return physical-space arrays."""
    with open(block_dir / "config_used.yaml") as f:
        bcfg = yaml.safe_load(f)

    n_folds = bcfg.get("n_folds", cfg_main.get("n_folds", 5))

    raw_blk = _slice_raw_to_block(
        raw_full, block_idx, block_size, block_overlap,
    )

    folds = kfold_indices(raw_blk["n_samples"], n_folds,
                          bcfg.get("random_seed", 42))
    branch_names = list(raw_blk["branch_names"])

    all_val_idx = []
    all_pred_phys = []
    all_target_phys = []
    all_branch_phys = {n: [] for n in branch_names}

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        fold_dir = block_dir / f"fold_{fold_i}"
        if not fold_dir.exists():
            continue

        ckpt = torch.load(fold_dir / "best_model.pt", map_location=device,
                           weights_only=False)
        model, fold_cfg = _load_model_from_checkpoint(
            ckpt, bcfg, raw_blk["n_levels"], device)
        scalers = _load_scalers(fold_dir / "scalers.npz", fold_cfg)

        data = make_dataloaders(raw_blk, train_idx, val_idx,
                                bcfg.get("batch_size", 64))

        pred_n, target_n = _predict(model, data["val_loader"], device)
        all_pred_phys.append(scalers["y"].inverse_transform(pred_n))
        all_target_phys.append(scalers["y"].inverse_transform(target_n))
        all_val_idx.append(val_idx)

        bp_n = _predict_branches(model, data["val_loader"], device)
        for name in branch_names:
            all_branch_phys[name].append(
                scalers["y"].inverse_transform(bp_n[name]))

    pred_phys = np.concatenate(all_pred_phys)
    target_phys = np.concatenate(all_target_phys)
    val_idx_all = np.concatenate(all_val_idx)
    branch_phys = {n: np.concatenate(v) for n, v in all_branch_phys.items()}

    return pred_phys, target_phys, val_idx_all, branch_phys


# ── Plotting ────────────────────────────────────────────────────────────────


def plot_training_curves(model_dir, n_folds, fig_dir, blocks=None):
    """Plot training curves. For blocked models, show one subplot per block."""
    if blocks is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for fold_i in range(n_folds):
            h = np.load(model_dir / f"fold_{fold_i}" / "training_history.npz")
            alpha = 0.3 if n_folds > 1 else 1.0
            ax.semilogy(h["train_loss"], color="C0", alpha=alpha,
                         label="Train" if fold_i == 0 else None)
            ax.semilogy(h["val_loss"], color="C1", alpha=alpha,
                         label="Validation" if fold_i == 0 else None)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (normalised)")
        ax.legend()
        ax.set_title(f"Training curves ({n_folds}-fold)")
    else:
        n_blocks = len(blocks)
        ncols = min(3, n_blocks)
        nrows = (n_blocks + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        for b_idx in range(n_blocks):
            ax = axes[b_idx // ncols][b_idx % ncols]
            bdir = model_dir / f"block_{b_idx}"
            for fold_i in range(n_folds):
                hpath = bdir / f"fold_{fold_i}" / "training_history.npz"
                if not hpath.exists():
                    continue
                h = np.load(hpath)
                alpha = 0.4 if n_folds > 1 else 1.0
                ax.semilogy(h["train_loss"], color="C0", alpha=alpha,
                             label="Train" if fold_i == 0 else None)
                ax.semilogy(h["val_loss"], color="C1", alpha=alpha,
                             label="Validation" if fold_i == 0 else None)
            bstart, bend = blocks[b_idx]
            ax.set_title(f"Block {b_idx} [{bstart},{bend})", fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            if b_idx == 0:
                ax.legend(fontsize=7)
        for idx in range(n_blocks, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle(f"Training curves ({n_blocks} blocks, {n_folds}-fold)",
                     y=1.02)

    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'training_curves.png'}")


def plot_bias_profiles(rmse, r2, n_levels, fig_dir, pressure_pa=None,
                       var_labels=("T bias", "Q bias")):
    yvals, ylabel = _y_axis(pressure_pa, n_levels)
    fig, axes = plt.subplots(1, len(var_labels), figsize=(5 * len(var_labels), 6))
    if len(var_labels) == 1:
        axes = [axes]
    for i, (label, ax) in enumerate(zip(var_labels, axes)):
        ax_r2 = ax.twiny()
        ax.plot(rmse[i], yvals, "b-", label="RMSE")
        ax_r2.plot(r2[i], yvals, "r--", label="R²")
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("RMSE (physical units)", color="b")
        ax_r2.set_xlabel("R²", color="r")
        ax.set_title(label)
        ax.invert_yaxis()
    fig.suptitle("Validation bias profiles (all folds)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "bias_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'bias_profiles.png'}")


def plot_branch_contributions(branch_means, branch_names, n_levels, fig_dir,
                              n_output_vars=2, pressure_pa=None,
                              var_labels=("T bias", "Q bias"),
                              target_mean=None, pred_mean=None,
                              branch_weights=None):
    """Mean predicted bias contribution per branch (physical units).

    When ``branch_weights`` is provided, each branch profile is scaled by
    its weight so the plotted profiles reflect the actual model output.
    ``pred_mean`` (the actual weighted prediction mean) is used for the
    sum line when available.
    """
    yvals, ylabel = _y_axis(pressure_pa, n_levels)
    fig, axes = plt.subplots(1, len(var_labels), figsize=(6 * len(var_labels), 7))
    if len(var_labels) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_names), 3)))
    for i, (vlabel, ax) in enumerate(zip(var_labels, axes)):
        if target_mean is not None:
            actual = target_mean.reshape(n_output_vars, n_levels)[i]
            ax.plot(actual, yvals, color="k", linewidth=2.2,
                    label="Actual (target)", zorder=5)

        total = None
        for j, name in enumerate(branch_names):
            mean_profile = branch_means[name]
            profile = mean_profile.reshape(n_output_vars, n_levels)[i]
            w = branch_weights.get(name, 1.0) if branch_weights else 1.0
            weighted = profile * w
            label = branch_label(name)
            if branch_weights and abs(w - 1.0) > 1e-6:
                label += f" (w={w:.2f})"
            ax.plot(
                weighted, yvals,
                label=label,
                color=colors[j % len(colors)],
                linewidth=1.1,
            )
            total = weighted if total is None else total + weighted

        if pred_mean is not None:
            sum_profile = pred_mean.reshape(n_output_vars, n_levels)[i]
        else:
            sum_profile = total
        ax.plot(sum_profile, yvals, "k--", linewidth=2.2,
                label="Prediction (weighted sum)")
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Mean contribution (physical units)")
        ax.set_title(vlabel)
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Partitioned branch contributions vs actual target (validation mean)",
        y=1.02,
    )
    fig.tight_layout()
    out = fig_dir / "branch_contributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def compute_branch_shapley(branch_phys, branch_names, n_levels, n_output_vars):
    """Compute branch-level Shapley importance profiles.

    For an additive model (PANN), the Shapley value of each branch is
    ``phi_b(x) = f_b(x) - E[f_b]``.  Returns a dict mapping branch name
    to the mean absolute Shapley profile of shape ``(n_output_vars, n_levels)``.
    """
    result = {}
    for name in branch_names:
        vals = branch_phys[name]
        phi = vals - vals.mean(axis=0, keepdims=True)
        phi_3d = np.abs(phi).reshape(phi.shape[0], n_output_vars, n_levels)
        result[name] = phi_3d.mean(axis=0)
    return result


def plot_branch_shapley_profiles(shapley_profiles, branch_names, n_levels,
                                 fig_dir, n_output_vars=1, pressure_pa=None,
                                 var_labels=("Q dQ",)):
    """Vertical profile of mean |Shapley| per branch (importance decomposition).

    Left panel: absolute importance lines per branch.
    Right panel: fractional importance (stacked area) showing relative
    contribution at each level.
    """
    yvals, ylabel = _y_axis(pressure_pa, n_levels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_names), 3)))

    for i, vlabel in enumerate(var_labels):
        fig, (ax_abs, ax_frac) = plt.subplots(
            1, 2, figsize=(11, 6), sharey=True,
        )

        profiles = {}
        for j, name in enumerate(branch_names):
            prof = shapley_profiles[name][i]
            profiles[name] = prof
            ax_abs.plot(
                prof, yvals,
                label=branch_label(name),
                color=colors[j % len(colors)],
                linewidth=1.5,
            )

        ax_abs.set_xlabel(r"Mean $|\phi|$ (physical units)")
        ax_abs.set_ylabel(ylabel)
        ax_abs.set_title(f"Branch Shapley importance — {vlabel}")
        ax_abs.invert_yaxis()
        ax_abs.legend(fontsize=8, loc="best")
        ax_abs.axvline(0, color="k", linewidth=0.5, alpha=0.3)

        total = sum(profiles.values())
        safe_total = np.where(total > 1e-30, total, 1.0)
        cumulative = np.zeros(n_levels)
        for j, name in enumerate(branch_names):
            frac = profiles[name] / safe_total
            ax_frac.fill_betweenx(
                yvals, cumulative, cumulative + frac,
                label=branch_label(name),
                color=colors[j % len(colors)],
                alpha=0.7,
            )
            cumulative = cumulative + frac

        ax_frac.set_xlabel("Fraction of total importance")
        ax_frac.set_xlim(0, 1)
        ax_frac.set_title(f"Relative branch importance — {vlabel}")
        ax_frac.legend(fontsize=8, loc="best")

        fig.tight_layout()
        suffix = f"_{vlabel.replace(' ', '_').lower()}" if len(var_labels) > 1 else ""
        out = fig_dir / f"branch_shapley_profiles{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def plot_branch_contributions_weighted(branch_means, shapley_profiles,
                                       branch_names, n_levels, fig_dir,
                                       n_output_vars=1, pressure_pa=None,
                                       var_labels=("Q bias",)):
    """Branch contributions weighted by Shapley fractional importance.

    At each level the raw mean contribution is scaled by the fraction of
    total mean-|Shapley| attributed to that branch, so visually
    less-important branches are attenuated.
    """
    yvals, ylabel = _y_axis(pressure_pa, n_levels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_names), 3)))

    for i, vlabel in enumerate(var_labels):
        fig, ax = plt.subplots(figsize=(6, 7))

        shapley_stack = np.stack(
            [shapley_profiles[n][i] for n in branch_names], axis=0,
        )
        total_shapley = shapley_stack.sum(axis=0)
        safe_total = np.where(total_shapley > 1e-30, total_shapley, 1.0)

        weighted_sum = None
        for j, name in enumerate(branch_names):
            frac = shapley_stack[j] / safe_total
            raw_profile = branch_means[name].reshape(n_output_vars, n_levels)[i]
            weighted = raw_profile * frac
            ax.plot(
                weighted, yvals,
                label=branch_label(name),
                color=colors[j % len(colors)],
                linewidth=1.3,
            )
            weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted

        ax.plot(weighted_sum, yvals, "k--", linewidth=2.2,
                label="sum (weighted)")
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Shapley-weighted mean contribution (physical units)")
        ax.set_title(vlabel)
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc="best")

        fig.suptitle(
            "Branch contributions weighted by Shapley importance (validation)",
            y=1.02,
        )
        fig.tight_layout()
        suffix = f"_{vlabel.replace(' ', '_').lower()}" if len(var_labels) > 1 else ""
        out = fig_dir / f"branch_contributions_shapley{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def plot_branch_vs_truth(branch_phys, branch_names, raw, n_levels,
                         val_idx_all, fig_dir, pressure_pa=None):
    """Compare predicted per-branch output against known truth.

    Supports both ``simple_linear_q`` (scalar weights) and
    ``simple_gaussian_q`` (per-level envelopes).

    Produces two figures:
    - **Profiles**: mean and std of predicted vs truth per level.
    - **Scatter**: per-sample scatter at three representative levels.
    """
    truth_weights = raw.get("truth_weights")       # linear mode
    truth_envelopes = raw.get("truth_envelopes")   # gaussian mode
    if truth_weights is None and truth_envelopes is None:
        return

    xs = raw["x_branch_list"]
    yvals, ylabel = _y_axis(pressure_pa, n_levels)

    # ── 1. Profile comparison (mean + std envelope) ──────────────────────
    fig, axes = plt.subplots(1, len(branch_names),
                             figsize=(6 * len(branch_names), 7))
    if len(branch_names) == 1:
        axes = [axes]

    for j, name in enumerate(branch_names):
        ax = axes[j]
        idx_b = list(branch_names).index(name)
        x_val = xs[idx_b][val_idx_all]

        if truth_envelopes is not None:
            env = truth_envelopes[name]  # shape (n_levels,)
            truth_all = env[None, :] * x_val
            label_str = f"Truth (envelope x Q)"
        else:
            w = truth_weights[name]
            truth_all = w * x_val
            label_str = f"Truth ({w:+.2g} x Q)"

        pred_all = branch_phys[name]

        truth_mean = truth_all.mean(axis=0)
        truth_std = truth_all.std(axis=0)
        pred_mean = pred_all.mean(axis=0)
        pred_std = pred_all.std(axis=0)

        ax.plot(truth_mean, yvals, "k-", linewidth=2.0, label=label_str)
        ax.fill_betweenx(yvals, truth_mean - truth_std, truth_mean + truth_std,
                         color="k", alpha=0.1, label="Truth \u00b11\u03c3")
        ax.plot(pred_mean, yvals, "C0-", linewidth=1.4, label="Predicted mean")
        ax.fill_betweenx(yvals, pred_mean - pred_std, pred_mean + pred_std,
                         color="C0", alpha=0.15, label="Predicted \u00b11\u03c3")
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Q bias contribution (physical units)")
        ax.set_title(f"{branch_label(name)} branch")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Branch attribution: predicted vs planted truth (validation)",
                 y=1.02)
    fig.tight_layout()
    out = fig_dir / "branch_vs_truth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

    # ── 1b. Envelope shape plot (gaussian mode only) ─────────────────────
    if truth_envelopes is not None:
        fig, ax = plt.subplots(figsize=(5, 6))
        for name in branch_names:
            ax.plot(truth_envelopes[name], yvals, label=branch_label(name),
                    linewidth=1.5)
        ax.set_xlabel("Envelope weight (scale x Gaussian)")
        ax.set_ylabel(ylabel)
        ax.set_title("Planted Gaussian envelopes per branch")
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        out = fig_dir / "truth_envelopes.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")

    # ── 2. Per-sample scatter at representative levels ───────────────────
    if pressure_pa is not None:
        scatter_levels = [
            int(np.argmin(np.abs(pressure_pa - 850e2))),
            int(np.argmin(np.abs(pressure_pa - 500e2))),
            int(np.argmin(np.abs(pressure_pa - 200e2))),
        ]
        scatter_labels = ["850 hPa", "500 hPa", "200 hPa"]
    else:
        scatter_levels = [n_levels - 5, n_levels // 2, 5]
        scatter_labels = [f"level {k}" for k in scatter_levels]

    fig, axes = plt.subplots(len(branch_names), len(scatter_levels),
                             figsize=(5 * len(scatter_levels),
                                      4.5 * len(branch_names)))
    if len(branch_names) == 1:
        axes = axes[np.newaxis, :]

    for j, name in enumerate(branch_names):
        idx_b = list(branch_names).index(name)
        x_val = xs[idx_b][val_idx_all]

        if truth_envelopes is not None:
            env = truth_envelopes[name]
            truth_all = env[None, :] * x_val
        else:
            truth_all = truth_weights[name] * x_val

        pred_all = branch_phys[name]

        for k, (lev_idx, lev_label) in enumerate(
                zip(scatter_levels, scatter_labels)):
            ax = axes[j, k]
            t = truth_all[:, lev_idx]
            p = pred_all[:, lev_idx]
            ax.scatter(t, p, alpha=0.3, s=4, rasterized=True)
            lim = max(np.abs(np.nanpercentile(t, [1, 99])).max(),
                      np.abs(np.nanpercentile(p, [1, 99])).max()) * 1.3
            if lim < 1e-30:
                lim = 1.0
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axline((0, 0), slope=1, color="r", ls="--", alpha=0.6,
                       label="1:1")
            corr = np.corrcoef(t, p)[0, 1] if np.std(t) > 0 else np.nan
            ax.text(0.04, 0.96, f"r = {corr:.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            ax.set_xlabel("Truth")
            ax.set_ylabel("Predicted branch")
            ax.set_title(f"{branch_label(name)} @ {lev_label}")
            ax.grid(True, alpha=0.15)

    fig.suptitle("Per-sample branch scatter: predicted vs truth", y=1.02)
    fig.tight_layout()
    out = fig_dir / "branch_scatter_truth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_corrected_tendency_scatter(nc_path, val_idx_all, pred_phys_all,
                                    n_levels, fig_dir):
    """Scatter: column-mean actual vs bias-corrected tendency (all folds)."""
    ds = xr.open_dataset(nc_path)
    if n_levels != ds.sizes["level"]:
        ds = ds.isel(level=slice(0, n_levels))

    ph = ds["pressure_h_initial"].values[val_idx_all]
    dp = np.diff(ph, axis=1)

    def column_mean(field, dp):
        return np.nansum(field * dp, axis=1) / np.nansum(dp, axis=1)

    T_actual = ds["T_tendency_actual"].values[val_idx_all]
    T_total  = ds["T_tendency_total"].values[val_idx_all]
    Q_actual = ds["Q_tendency_actual"].values[val_idx_all]
    Q_total  = ds["Q_tendency_total"].values[val_idx_all]

    pred_T_bias = pred_phys_all[:, :n_levels]
    pred_Q_bias = pred_phys_all[:, n_levels:]

    T_corrected = T_total - pred_T_bias
    Q_corrected = Q_total - pred_Q_bias

    T_actual_col    = column_mean(T_actual, dp) * 3600
    T_corrected_col = column_mean(T_corrected, dp) * 3600
    Q_actual_col    = column_mean(Q_actual, dp) * 3600 * 1000
    Q_corrected_col = column_mean(Q_corrected, dp) * 3600 * 1000

    valid_T = np.isfinite(T_actual_col) & np.isfinite(T_corrected_col)
    valid_Q = np.isfinite(Q_actual_col) & np.isfinite(Q_corrected_col)

    EPS = 1e-30
    has_conv  = np.any(np.abs(ds["T_tendency_conv"].values[val_idx_all]) > EPS, axis=1)
    has_cloud = np.any(np.abs(ds["T_tendency_cloud"].values[val_idx_all]) > EPS, axis=1)
    ds.close()

    CAT_MASKS = [
        ("Neither",    (~has_conv) & (~has_cloud)),
        ("Cloud only", (~has_conv) &  (has_cloud)),
        ("Both",        (has_conv) &  (has_cloud)),
    ]
    CAT_COLORS = {"Neither": "skyblue", "Cloud only": "grey", "Both": "midnightblue"}

    def _scatter_by_regime(ax, x_all, y_all, valid):
        for label, cat_mask in CAT_MASKS:
            m = cat_mask & valid
            if m.sum() == 0:
                continue
            ax.scatter(x_all[m], y_all[m], alpha=0.45, s=6,
                       color=CAT_COLORS[label], label=f"{label} ({m.sum()})",
                       rasterized=True, zorder=2)

    def _dress_ax(ax, x, y, xlabel, ylabel, title):
        lim = max(np.abs(np.percentile(x, [1, 99])).max(),
                  np.abs(np.percentile(y, [1, 99])).max()) * 1.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axline((0, 0), slope=1, color="r", linestyle="--", alpha=0.6, label="1:1")
        ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
        ax.axvline(0, color="k", linewidth=0.4, alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.04, 0.96, f"r = {corr:.3f}\nn = {len(x)}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right", fontsize=8, markerscale=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    _scatter_by_regime(ax, T_actual_col, T_corrected_col, valid_T)
    _dress_ax(ax, T_actual_col[valid_T], T_corrected_col[valid_T],
              r"Actual $\langle dT/dt \rangle_{col}$  (K/hour)",
              r"Corrected $\langle dT/dt \rangle_{col}$  (K/hour)",
              "Column-Mean Temperature Tendency (bias-corrected)")

    ax = axes[1]
    _scatter_by_regime(ax, Q_actual_col, Q_corrected_col, valid_Q)
    _dress_ax(ax, Q_actual_col[valid_Q], Q_corrected_col[valid_Q],
              r"Actual $\langle dQ/dt \rangle_{col}$  (g kg$^{-1}$ hour$^{-1}$)",
              r"Corrected $\langle dQ/dt \rangle_{col}$  (g kg$^{-1}$ hour$^{-1}$)",
              "Column-Mean Humidity Tendency (bias-corrected)")

    fig.tight_layout()
    out = fig_dir / "tendency_comparison_corrected.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_pred_vs_actual_scatter(pred_phys, target_phys, n_levels, fig_dir,
                                n_output_vars=1, pressure_pa=None,
                                var_labels=("Q dQ",)):
    """Scatter of predicted vs actual dQ at key pressure levels."""
    if pressure_pa is not None:
        level_targets_hpa = [850, 700, 500, 300, 200]
        scatter_levels = []
        scatter_labels = []
        for hpa in level_targets_hpa:
            idx = int(np.argmin(np.abs(pressure_pa - hpa * 100)))
            scatter_levels.append(idx)
            actual_hpa = pressure_pa[idx] / 100
            scatter_labels.append(f"{actual_hpa:.0f} hPa")
    else:
        n = n_levels
        scatter_levels = [n - 2, 3 * n // 4, n // 2, n // 4, 2]
        scatter_labels = [f"level {k}" for k in scatter_levels]

    pred_3d = pred_phys.reshape(pred_phys.shape[0], n_output_vars, n_levels)
    tgt_3d = target_phys.reshape(target_phys.shape[0], n_output_vars, n_levels)

    for vi, vlabel in enumerate(var_labels):
        ncols = len(scatter_levels)
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.2))
        if ncols == 1:
            axes = [axes]

        for k, (lev_idx, lev_label) in enumerate(
                zip(scatter_levels, scatter_labels)):
            ax = axes[k]
            t = tgt_3d[:, vi, lev_idx]
            p = pred_3d[:, vi, lev_idx]

            ax.scatter(t, p, alpha=0.25, s=4, color="C0", rasterized=True)

            finite = np.isfinite(t) & np.isfinite(p)
            if finite.sum() > 2:
                lim = max(
                    np.abs(np.nanpercentile(t[finite], [1, 99])).max(),
                    np.abs(np.nanpercentile(p[finite], [1, 99])).max(),
                ) * 1.3
            else:
                lim = 1.0
            if lim < 1e-30:
                lim = 1.0
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axline((0, 0), slope=1, color="r", ls="--", alpha=0.6,
                       label="1:1")
            ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
            ax.axvline(0, color="k", linewidth=0.4, alpha=0.3)

            if finite.sum() > 2 and np.std(t[finite]) > 0:
                corr = np.corrcoef(t[finite], p[finite])[0, 1]
                rmse_val = np.sqrt(np.mean((p[finite] - t[finite]) ** 2))
            else:
                corr = np.nan
                rmse_val = np.nan
            ax.text(0.04, 0.96,
                    f"r = {corr:.3f}\nRMSE = {rmse_val:.2e}\nn = {finite.sum()}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            ax.set_xlabel("Actual (physical units)")
            ax.set_ylabel("Predicted (physical units)")
            ax.set_title(f"{vlabel} @ {lev_label}")
            ax.grid(True, alpha=0.15)

        fig.suptitle("Predicted vs actual by level (validation)", y=1.02)
        fig.tight_layout()
        suffix = f"_{vlabel.replace(' ', '_').lower()}" if len(var_labels) > 1 else ""
        out = fig_dir / f"pred_vs_actual_scatter{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Main ────────────────────────────────────────────────────────────────────

_SYNTHETIC_MODES = (TARGET_MODE_SIMPLE_LINEAR_Q, TARGET_MODE_SIMPLE_GAUSSIAN_Q)


def _detect_blocks(model_dir):
    """Return sorted list of block indices if blocked layout, else None."""
    block_dirs = sorted(model_dir.glob("block_*"))
    if not block_dirs:
        return None
    indices = []
    for d in block_dirs:
        try:
            indices.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(indices) if indices else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate PANN (k-fold)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--fig_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_folds", type=int, default=None,
                        help="Override number of folds to evaluate")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    block_indices = _detect_blocks(model_dir)
    is_blocked = block_indices is not None

    if is_blocked:
        with open(model_dir / "block_0" / "config_used.yaml") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(model_dir / "config_used.yaml") as f:
            cfg = yaml.safe_load(f)

    n_folds = args.n_folds or cfg.get("n_folds", 5)
    n_out = int(cfg.get("n_output_vars", 2))
    block_size = cfg.get("block_size")
    block_overlap = cfg.get("block_overlap")

    raw = load_raw_arrays(
        args.data,
        n_levels=cfg.get("n_levels"),
        level_start=cfg.get("level_start"),
        target_cfg=cfg.get("target"),
    )
    n_levels = raw["n_levels"]
    branch_names = list(raw["branch_names"])
    target_mode = raw.get("target_mode", "scm_minus_reference")
    pressure_pa = raw.get("pressure_pa")

    if is_blocked:
        print(f"Blocked evaluation: {len(block_indices)} blocks, "
              f"block_size={block_size}, overlap={block_overlap}")
        blocks = compute_blocks(n_levels, block_size, block_overlap)
        blend_weights = _build_blend_weights(blocks, n_levels)

        block_pred_list = []
        block_target_list = []
        block_branch_lists = {n: [] for n in branch_names}
        val_idx_all = None

        for b_idx in block_indices:
            block_dir = model_dir / f"block_{b_idx}"
            print(f"\nBlock {b_idx}: {block_dir}")
            pred_b, tgt_b, vidx_b, bp_b = _evaluate_one_block(
                block_dir, raw, cfg, device,
                b_idx, block_size, block_overlap,
            )
            block_pred_list.append(pred_b)
            block_target_list.append(tgt_b)
            for n in branch_names:
                block_branch_lists[n].append(bp_b[n])
            if val_idx_all is None:
                val_idx_all = vidx_b

        pred_phys = _stitch_blocks(block_pred_list, blocks, blend_weights,
                                   n_levels, n_out)
        target_phys = _stitch_blocks(block_target_list, blocks, blend_weights,
                                     n_levels, n_out)
        branch_phys = {}
        for n in branch_names:
            branch_phys[n] = _stitch_blocks(
                block_branch_lists[n], blocks, blend_weights, n_levels, n_out)

    else:
        folds = kfold_indices(raw["n_samples"], n_folds,
                              cfg.get("random_seed", 42))

        all_val_idx = []
        all_pred_phys = []
        all_target_phys = []
        all_branch_phys = {n: [] for n in branch_names}

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            if fold_i >= n_folds:
                break
            fold_dir = model_dir / f"fold_{fold_i}"
            print(f"Fold {fold_i}: loading {fold_dir}")

            ckpt = torch.load(fold_dir / "best_model.pt", map_location=device,
                               weights_only=False)
            model, fold_cfg = _load_model_from_checkpoint(
                ckpt, cfg, n_levels, device)
            scalers = _load_scalers(fold_dir / "scalers.npz", fold_cfg)
            data = make_dataloaders(raw, train_idx, val_idx, cfg["batch_size"])

            pred_n, target_n = _predict(model, data["val_loader"], device)
            all_pred_phys.append(scalers["y"].inverse_transform(pred_n))
            all_target_phys.append(scalers["y"].inverse_transform(target_n))
            all_val_idx.append(val_idx)

            bp_n = _predict_branches(model, data["val_loader"], device)
            for name in branch_names:
                all_branch_phys[name].append(
                    scalers["y"].inverse_transform(bp_n[name]))

        pred_phys = np.concatenate(all_pred_phys)
        target_phys = np.concatenate(all_target_phys)
        val_idx_all = np.concatenate(all_val_idx)
        branch_phys = {n: np.concatenate(v) for n, v in all_branch_phys.items()}
        blocks = None

    # ── Metrics ──────────────────────────────────────────────────────────
    rmse = rmse_per_level(pred_phys, target_phys, n_levels, n_output_vars=n_out)
    r2 = r2_per_level(pred_phys, target_phys, n_levels, n_output_vars=n_out)

    if n_out == 1:
        print(f"\nOverall RMSE  Q bias: {np.mean(rmse[0]):.4e}")
        print(f"Overall R\u00b2    Q bias: {np.nanmean(r2[0]):.4f}  "
              f"(valid at {np.sum(np.isfinite(r2[0]))}/{n_levels} levels)")
        var_labels = ("Q bias",)
    else:
        print(f"\nOverall RMSE  T: {np.mean(rmse[0]):.4e}  Q: {np.mean(rmse[1]):.4e}")
        print(f"Overall R\u00b2    T: {np.nanmean(r2[0]):.4f}    Q: {np.nanmean(r2[1]):.4f}")
        var_labels = ("T bias", "Q bias")

    branch_means_phys = {n: v.mean(axis=0) for n, v in branch_phys.items()}

    # ── Plots ────────────────────────────────────────────────────────────
    plot_training_curves(model_dir, n_folds, fig_dir, blocks=blocks)
    plot_bias_profiles(
        rmse, r2, n_levels, fig_dir, pressure_pa=pressure_pa,
        var_labels=var_labels,
    )
    target_mean_phys = target_phys.mean(axis=0)
    pred_mean_phys = pred_phys.mean(axis=0)

    # branch_phys is already weighted by branch_weights (applied in
    # _predict_branches), so no extra weight scaling is needed here.
    plot_branch_contributions(
        branch_means_phys, branch_names, n_levels, fig_dir,
        n_output_vars=n_out, pressure_pa=pressure_pa, var_labels=var_labels,
        target_mean=target_mean_phys, pred_mean=pred_mean_phys,
    )

    if target_mode in _SYNTHETIC_MODES:
        plot_branch_vs_truth(
            branch_phys, branch_names, raw, n_levels,
            val_idx_all, fig_dir, pressure_pa=pressure_pa,
        )
        print("  Skipping tendency scatter (synthetic target).")
    elif n_out == 2:
        plot_corrected_tendency_scatter(
            args.data, val_idx_all, pred_phys, n_levels, fig_dir,
        )
    else:
        plot_pred_vs_actual_scatter(
            pred_phys, target_phys, n_levels, fig_dir,
            n_output_vars=n_out, pressure_pa=pressure_pa, var_labels=var_labels,
        )

    # ── Branch Shapley importance profiles ────────────────────────────────
    if len(branch_names) >= 2:
        shapley_profiles = compute_branch_shapley(
            branch_phys, branch_names, n_levels, n_out,
        )
        plot_branch_shapley_profiles(
            shapley_profiles, branch_names, n_levels, fig_dir,
            n_output_vars=n_out, pressure_pa=pressure_pa, var_labels=var_labels,
        )
        plot_branch_contributions_weighted(
            branch_means_phys, shapley_profiles, branch_names, n_levels, fig_dir,
            n_output_vars=n_out, pressure_pa=pressure_pa, var_labels=var_labels,
        )
    else:
        shapley_profiles = {}

    # ── Save metrics ─────────────────────────────────────────────────────
    metrics_kw = dict(
        rmse=rmse,
        r2=r2,
        val_idx=val_idx_all,
        pred_phys=pred_phys,
        target_phys=target_phys,
    )
    for n in branch_names:
        metrics_kw[f"branch_mean_{n}"] = branch_means_phys[n]
    for n in branch_names:
        if n in shapley_profiles:
            metrics_kw[f"shapley_mean_abs_{n}"] = shapley_profiles[n]
    np.savez(fig_dir / "metrics.npz", **metrics_kw)
    (fig_dir / "eval_branch_names.txt").write_text("\n".join(branch_names) + "\n")
    print(f"Metrics saved to {fig_dir / 'metrics.npz'}")


if __name__ == "__main__":
    main()
