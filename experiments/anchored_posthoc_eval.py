"""Post-hoc anchoring of an already-trained (free-bias) PANN.

A free PANN  y = sum_b w_b NN_b(x_b)  is rewritten exactly as

    y = sum_b w_b [NN_b(x_b) - NN_b(x_off_b)]  +  sum_b w_b NN_b(x_off_b)
        `------ anchored branch f_b ------'      `--- global intercept ---'

with ``x_off_b`` the normalised image of a physically zero input.  Predictions
are unchanged; only the split between branches and the intercept changes
(see ``anchored_bias.md``).  No retraining.

Reads the models trained by the main pipeline (``output/models``, blocked or
not), and compares the per-branch mean contribution E[f_b] under

  * evaluate.py convention:  f_b * y_std + y_mean   (y_mean in every branch)
  * post-hoc anchored:       w_b (NN_b(x) - NN_b(x_off)) * y_std,
                             intercept = sum_b w_b NN_b(x_off) * y_std + y_mean

against the planted truth (``simple_gaussian_q``).  Variance Shapley
E|f_b - E f_b| is identical under both and is drawn as a control.

Run:
    python experiments/anchored_posthoc_eval.py \
        --model_dir output/models --out_dir output/experiments/anchored_posthoc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset import (  # noqa: E402
    _slice_raw_to_block,
    compute_blocks,
    kfold_indices,
    load_raw_arrays,
    make_dataloaders,
)
from evaluate import (  # noqa: E402
    _build_blend_weights,
    _detect_blocks,
    _load_model_from_checkpoint,
    _load_scalers,
    _stitch_blocks,
    compute_truth_branch_phys,
)

BRANCH_COLORS = {"conv": ("#0b4f9c", "#9ecae1"), "cloud": ("#d94801", "#fdbe85")}


def _anchored_fold(model, scalers, data, branch_names):
    """Per-branch evaluate.py and anchored outputs (physical) for one fold."""
    model.eval()
    y_std, y_mean = scalers["y"].std, scalers["y"].mean
    raw_out = {n: [] for n in branch_names}
    preds = []
    with torch.no_grad():
        for *xb, _ in data["val_loader"]:
            preds.append(model(*xb).numpy())
            for n, x in zip(branch_names, xb):
                raw_out[n].append(model.branches[n](x).numpy())
        off_out = {}
        for n in branch_names:
            n_in = data["val_loader"].dataset.xs[branch_names.index(n)].shape[1]
            x_off = scalers[n].transform(np.zeros((1, n_in))).astype(np.float32)
            off_out[n] = model.branches[n](torch.as_tensor(x_off)).numpy()[0]

    evalpy, anchored = {}, {}
    intercept = y_mean.copy()
    for n in branch_names:
        active = model.branch_mask.get(n, True)
        w = model.branch_weights.get(n, 1.0) if active else 0.0
        f = w * np.concatenate(raw_out[n])
        evalpy[n] = f * y_std + y_mean
        anchored[n] = (f - w * off_out[n][None, :]) * y_std
        intercept = intercept + w * off_out[n] * y_std
    pred = np.concatenate(preds) * y_std + y_mean
    return evalpy, anchored, intercept, pred


def _eval_unit(unit_dir, raw, cfg, branch_names):
    """All folds of one block (or of the unblocked model)."""
    folds = kfold_indices(raw["n_samples"], cfg.get("n_folds", 5),
                          cfg.get("random_seed", 42))
    ev = {n: [] for n in branch_names}
    an = {n: [] for n in branch_names}
    icpt, val_idx, preds = [], [], []
    for fold_i, (train_idx, v_idx) in enumerate(folds):
        fold_dir = unit_dir / f"fold_{fold_i}"
        if not fold_dir.exists():
            continue
        ckpt = torch.load(fold_dir / "best_model.pt", map_location="cpu",
                          weights_only=False)
        model, fold_cfg = _load_model_from_checkpoint(
            ckpt, cfg, raw["n_levels"], torch.device("cpu"))
        scalers = _load_scalers(fold_dir / "scalers.npz", fold_cfg)
        data = make_dataloaders(raw, train_idx, v_idx, cfg.get("batch_size", 64))
        e, a, c, p = _anchored_fold(model, scalers, data, branch_names)
        for n in branch_names:
            ev[n].append(e[n])
            an[n].append(a[n])
        icpt.append(np.broadcast_to(c, (len(v_idx), c.size)))
        preds.append(p)
        val_idx.append(v_idx)
    cat = lambda d: {n: np.concatenate(v) for n, v in d.items()}  # noqa: E731
    return (cat(ev), cat(an), np.concatenate(icpt), np.concatenate(val_idx),
            np.concatenate(preds))


def collect(model_dir, data_path):
    blocks_idx = _detect_blocks(model_dir)
    cfg_path = (model_dir / "block_0" if blocks_idx is not None
                else model_dir) / "config_used.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    raw = load_raw_arrays(data_path, n_levels=cfg.get("n_levels"),
                          level_start=cfg.get("level_start"),
                          target_cfg=cfg.get("target"))
    branch_names = list(raw["branch_names"])
    n_levels, n_out = raw["n_levels"], raw["n_output_vars"]

    if blocks_idx is None:
        ev, an, icpt, val_idx, pred = _eval_unit(model_dir, raw, cfg,
                                                 branch_names)
    else:
        blocks = compute_blocks(n_levels, cfg["block_size"], cfg["block_overlap"])
        blend = _build_blend_weights(blocks, n_levels)
        parts = {"ev": {n: [] for n in branch_names},
                 "an": {n: [] for n in branch_names}, "icpt": [], "pred": []}
        val_idx = None
        for b in blocks_idx:
            print(f"  block {b}", flush=True)
            raw_b = _slice_raw_to_block(raw, b, cfg["block_size"],
                                        cfg["block_overlap"])
            with open(model_dir / f"block_{b}" / "config_used.yaml") as f:
                bcfg = yaml.safe_load(f)
            e, a, c, v, p = _eval_unit(model_dir / f"block_{b}", raw_b, bcfg,
                                       branch_names)
            for n in branch_names:
                parts["ev"][n].append(e[n])
                parts["an"][n].append(a[n])
            parts["icpt"].append(c)
            parts["pred"].append(p)
            val_idx = v if val_idx is None else val_idx
        stitch = lambda lst: _stitch_blocks(lst, blocks, blend, n_levels, n_out)  # noqa: E731
        ev = {n: stitch(v) for n, v in parts["ev"].items()}
        an = {n: stitch(v) for n, v in parts["an"].items()}
        icpt = stitch(parts["icpt"])
        pred = stitch(parts["pred"])

    truth = compute_truth_branch_phys(raw, branch_names, val_idx)
    return dict(branch_names=branch_names, pressure_pa=raw["pressure_pa"],
                evalpy=ev, anchored=an, intercept=icpt, truth=truth,
                y=raw["y"][val_idx], pred=pred)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def report(res):
    names, t = res["branch_names"], res["truth"]
    print("\nRelative L2 error of E[f_b] vs planted truth")
    print(f"  {'convention':<22}" + "".join(f"{n:>10}" for n in names)
          + f"{'|intercept|/|E y|':>20}")
    y_mean = res["y"].mean(0)
    for key, label in (("evalpy", "evaluate.py"), ("anchored", "post-hoc anchored")):
        errs = [_rel(res[key][n].mean(0), t[n].mean(0)) for n in names]
        icpt = (np.linalg.norm(res["intercept"].mean(0)) / np.linalg.norm(y_mean)
                if key == "anchored" else float("nan"))
        print(f"  {label:<22}" + "".join(f"{e:>10.3f}" for e in errs)
              + f"{icpt:>20.3f}")
    pred = res["intercept"] + sum(res["anchored"][n] for n in names)
    err = np.abs(pred - res["pred"]).max() / res["y"].std()
    print(f"\n  check: anchored sum vs model prediction, max abs diff = "
          f"{err:.1e} x std(y)")


def plot(res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names, t = res["branch_names"], res["truth"]
    p = res["pressure_pa"] / 100.0
    scale = 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), sharey=True)
    panels = (("evalpy", "(a) mean, evaluate.py"),
              ("anchored", "(b) mean, post-hoc anchored"),
              (None, "(c) variance Shapley (same for both)"))
    for ax, (key, title) in zip(axes, panels):
        for n in names:
            dark, pale = BRANCH_COLORS.get(n, ("k", "0.7"))
            if key is None:
                tv = np.abs(t[n] - t[n].mean(0)).mean(0)
                f = res["anchored"][n]
                pv = np.abs(f - f.mean(0)).mean(0)
            else:
                tv, pv = t[n].mean(0), res[key][n].mean(0)
            ax.plot(tv / scale, p, color=pale, lw=5, zorder=1)
            ax.plot(pv / scale, p, color=dark, lw=1.3, zorder=3)
        if key == "anchored":
            ax.plot(res["intercept"].mean(0) / scale, p, color="0.3", lw=1.0,
                    ls="--", zorder=2)
        if key is not None:
            ax.axvline(0, color="k", lw=0.5, alpha=0.3)
            ax.set_xlabel(r"Mean $E[f_b]$, Q bias [$10^{-9}$]")
        else:
            ax.set_xlim(left=0)
            ax.set_xlabel(r"Mean $|f_b - E f_b|$ [$10^{-9}$]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Pressure (hPa)")
    axes[0].set_ylim(p.max(), p.min())
    handles = []
    for n in names:
        dark, pale = BRANCH_COLORS.get(n, ("k", "0.7"))
        handles += [Line2D([], [], color=pale, lw=5, label=f"{n} truth"),
                    Line2D([], [], color=dark, lw=1.3, label=f"{n} PANN")]
    handles.append(Line2D([], [], color="0.3", lw=1, ls="--",
                          label="global intercept (truth 0)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_dir", default=str(REPO / "output" / "models"))
    ap.add_argument("--data", default=str(REPO / "data" /
                                          "training_dataset_partitioned.nc"))
    ap.add_argument("--out_dir", default=str(REPO / "output" / "experiments" /
                                             "anchored_posthoc"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = collect(Path(args.model_dir), args.data)
    if res["truth"] is None:
        raise ValueError("models were not trained on a planted-truth target")
    report(res)
    plot(res, out_dir / "anchoredPosthoc_meanContribution.png")


if __name__ == "__main__":
    main()
