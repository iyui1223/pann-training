"""Does tying each branch's bias to its input fix mean-contribution attribution?

In the PANN every branch is ``f_b(x_b) = NN_b(x_b)`` and only the SUM of the
branches enters the loss.  A constant can therefore be moved from one branch
to another without changing the prediction, so the per-branch mean
``E[f_b]`` (``branch_contributions_shapley.png``) is not identifiable, while
the variance-based Shapley ``|phi_b| = |f_b - E[f_b]|``
(``branch_shapley_profiles.png``) cancels that constant and is.

Three branch variants are trained on the same planted ``simple_gaussian_q``
truth ``y = env_conv * x_conv + env_cloud * x_cloud`` (zero output when a
process is off).  All variants carry one shared, unattributed per-level
intercept ``c`` so they differ ONLY in how the branch bias is treated:

  free      f_b = NN_b(x)                      current PANN; bias always on
  gated     f_b = 1[x_b != 0] * NN_b(x)        output (incl. bias) switched
                                               off when the input column is
                                               entirely zero (process off)
  anchored  f_b = NN_b(x) - NN_b(x_off)        f_b(off) = 0 exactly; the
                                               free constant cancels for
                                               every sample, not only the
                                               "off" ones

``x_off`` is the normalised image of a physically zero input, i.e.
``(0 - mean) / std`` -- the z-scoring means "off" is NOT zero inside the
network.  Branch contributions are converted to physical units by scaling
only (``f_b * y_std``); ``y_mean`` belongs to the intercept.  The planted
truth has zero intercept, so a correct decomposition should drive
``c * y_std + y_mean`` to ~0.

Each variant is trained for several seeds on a fixed split (fold 0): seed
spread of ``E[f_b]`` directly measures how un-identified the bias is.
Full-column training (no level blocking) and no branch reweighting, so the
bias treatment is the only difference.

Run:
    python experiments/zero_anchored_bias.py --out_dir output/experiments/zero_anchored_bias
    python experiments/zero_anchored_bias.py --out_dir ... --plot_only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset import kfold_indices, load_raw_arrays, make_dataloaders  # noqa: E402
from model import PANN  # noqa: E402
from train import _make_loss_fn, make_level_weights  # noqa: E402

VARIANTS = ("free", "gated", "anchored")
BRANCH_COLORS = {"conv": ("#0b4f9c", "#9ecae1"), "cloud": ("#d94801", "#fdbe85")}


class BiasModePANN(nn.Module):
    """PANN wrapper adding a global intercept and a branch-bias treatment."""

    def __init__(self, pann, x_off, mode):
        super().__init__()
        if mode not in VARIANTS:
            raise ValueError(f"unknown mode {mode!r}")
        self.pann = pann
        self.mode = mode
        self.branch_names = pann.branch_names
        for name in self.branch_names:
            self.register_buffer(f"x_off_{name}",
                                 torch.as_tensor(x_off[name], dtype=torch.float32))
        self.intercept = nn.Parameter(
            torch.zeros(pann.n_output_vars * pann.n_levels))

    def branch_outputs(self, *xs):
        outs = {}
        for name, x in zip(self.branch_names, xs):
            net = self.pann.branches[name]
            x_off = getattr(self, f"x_off_{name}")
            f = net(x)
            if self.mode == "anchored":
                f = f - net(x_off[None, :])
            elif self.mode == "gated":
                # A physical zero maps to exactly x_off (same float64 transform,
                # same float32 cast), so exact comparison is safe.
                active = (x != x_off[None, :]).any(dim=1, keepdim=True)
                f = f * active.to(f.dtype)
            outs[name] = f
        return outs

    def forward(self, *xs):
        return self.intercept + sum(self.branch_outputs(*xs).values())


def build_model(cfg, raw, scalers, mode):
    branches = [(n, raw["x_branch_list"][i].shape[1])
                for i, n in enumerate(raw["branch_names"])]
    pann = PANN(
        n_levels=raw["n_levels"],
        hidden_dim=cfg["hidden_dim"],
        branches=branches,
        architecture=cfg.get("architecture", "conv1d"),
        n_hidden_layers=cfg.get("n_hidden_layers", 4),
        bottleneck_dim=cfg.get("bottleneck_dim", 10),
        dropout=cfg.get("dropout", 0.0),
        n_output_vars=raw["n_output_vars"],
        conv_channels=cfg.get("conv_channels"),
    )
    x_off = {
        n: scalers[n].transform(np.zeros((1, raw["x_branch_list"][i].shape[1])))
        .astype(np.float32)[0]
        for i, n in enumerate(raw["branch_names"])
    }
    return BiasModePANN(pann, x_off, mode)


def train(model, data, cfg, loss_fn, epochs, patience):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for *xb, y in data["train_loader"]:
            opt.zero_grad()
            loss = loss_fn(model(*xb), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = sum(loss_fn(model(*xb), y).item() * y.size(0)
                      for *xb, y in data["val_loader"])
        val /= len(data["val_loader"].dataset)
        if val < best:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1:3d}  val_loss={val:.4e}  best={best:.4e}",
                  flush=True)
    model.load_state_dict(best_state)
    return best


def predict_physical(model, data, scaler_y):
    """Per-branch contributions and intercept in physical units (val set)."""
    model.eval()
    parts = {n: [] for n in model.branch_names}
    with torch.no_grad():
        for *xb, _ in data["val_loader"]:
            for n, f in model.branch_outputs(*xb).items():
                parts[n].append(f.numpy())
        c = model.intercept.numpy()
    std, mean = scaler_y.std, scaler_y.mean
    branch_phys = {n: np.concatenate(v) * std for n, v in parts.items()}
    intercept_phys = c * std + mean
    return branch_phys, intercept_phys


def truth_physical(raw, val_idx):
    return {n: raw["truth_envelopes"][n][None, :] * raw["x_branch_list"][i][val_idx]
            for i, n in enumerate(raw["branch_names"])}


def run_all(args, cfg):
    t0 = time.time()
    raw = load_raw_arrays(args.data, n_levels=cfg.get("n_levels"),
                          level_start=cfg.get("level_start"),
                          target_cfg=cfg["target"])
    print(f"Loaded {raw['n_samples']} samples x {raw['n_levels']} levels "
          f"in {time.time() - t0:.0f}s", flush=True)
    train_idx, val_idx = kfold_indices(raw["n_samples"], cfg["n_folds"],
                                       cfg["random_seed"])[0]
    level_w = (make_level_weights(raw["pressure_pa"], raw["n_output_vars"])
               if cfg.get("level_weight", "pressure") == "pressure" else None)
    loss_fn = _make_loss_fn(level_w, torch.device("cpu"))

    out_dir = Path(args.out_dir)
    np.savez(out_dir / "truth.npz", pressure_pa=raw["pressure_pa"],
             val_idx=val_idx, y_val=raw["y"][val_idx],
             **{f"truth_{n}": v for n, v in truth_physical(raw, val_idx).items()})

    for mode in args.variants:
        for seed in args.seeds:
            tag = f"{mode}_seed{seed}"
            print(f"\n=== {tag} ===", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            data = make_dataloaders(raw, train_idx, val_idx, cfg["batch_size"])
            model = build_model(cfg, raw, data["scalers"], mode)
            t1 = time.time()
            best = train(model, data, cfg, loss_fn, args.epochs, args.patience)
            branch_phys, intercept_phys = predict_physical(
                model, data, data["scalers"]["y"])
            np.savez(out_dir / f"run_{tag}.npz", best_val_loss=best,
                     intercept=intercept_phys,
                     **{f"branch_{n}": v for n, v in branch_phys.items()})
            print(f"  done in {time.time() - t1:.0f}s  best_val_loss={best:.4e}",
                  flush=True)


# ── Analysis ────────────────────────────────────────────────────────────────


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def load_runs(out_dir):
    truth = dict(np.load(out_dir / "truth.npz"))
    names = [k[len("truth_"):] for k in truth if k.startswith("truth_")]
    runs = {}
    for mode in VARIANTS:
        files = sorted(out_dir.glob(f"run_{mode}_seed*.npz"))
        if files:
            runs[mode] = [dict(np.load(f)) for f in files]
    return truth, names, runs


def summarise(truth, names, runs):
    y = truth["y_val"]
    t_mean = {n: truth[f"truth_{n}"].mean(0) for n in names}
    t_phi = {n: np.abs(truth[f"truth_{n}"] - t_mean[n]).mean(0) for n in names}
    rows = []
    for mode, seeds in runs.items():
        for r in seeds:
            pred = r["intercept"] + sum(r[f"branch_{n}"] for n in names)
            ss_res = ((pred - y) ** 2).sum(0)
            ss_tot = ((y - y.mean(0)) ** 2).sum(0)
            ok = ss_tot > 0
            row = {"variant": mode,
                   "r2_mean": float(np.mean(1 - ss_res[ok] / ss_tot[ok])),
                   "intercept_rel": float(np.linalg.norm(r["intercept"])
                                          / np.linalg.norm(y.mean(0)))}
            for n in names:
                f = r[f"branch_{n}"]
                row[f"mean_err_{n}"] = _rel(f.mean(0), t_mean[n])
                # Current evaluate.py adds y_mean back to every branch.
                row[f"mean_err_evalpy_{n}"] = _rel(f.mean(0) + y.mean(0), t_mean[n])
                row[f"phi_err_{n}"] = _rel(np.abs(f - f.mean(0)).mean(0), t_phi[n])
            rows.append(row)
    return rows


def print_table(rows, names):
    keys = (["r2_mean", "intercept_rel"]
            + [f"mean_err_{n}" for n in names]
            + [f"phi_err_{n}" for n in names])
    print("\nRelative L2 error vs planted truth (mean +- std over seeds)")
    print("  mean_err = E[f_b] profile (bias-sensitive); "
          "phi_err = mean|f_b - E f_b| (bias-free)")
    print("  intercept_rel = |c_phys| / |E[y]|  (truth: 0)\n")
    print(f"  {'variant':<10}" + "".join(f"{k:>18}" for k in keys))
    for mode in VARIANTS:
        sel = [r for r in rows if r["variant"] == mode]
        if not sel:
            continue
        cells = []
        for k in keys:
            v = np.array([r[k] for r in sel])
            cells.append(f"{v.mean():>11.3f}+-{v.std():<5.3f}")
        print(f"  {mode:<10}" + "".join(cells))
    free = [r for r in rows if r["variant"] == "free"]
    if free:
        print("\n  free variant with evaluate.py convention (y_mean added to "
              "every branch): "
              + ", ".join(f"mean_err_{n}="
                          f"{np.mean([r[f'mean_err_evalpy_{n}'] for r in free]):.3f}"
                          for n in names))


def plot(truth, names, runs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = truth["pressure_pa"] / 100.0
    modes = [m for m in VARIANTS if m in runs]
    scale = 1e-9
    fig, axes = plt.subplots(2, len(modes), figsize=(3.3 * len(modes), 8.2),
                             sharey=True, sharex="row", squeeze=False)
    titles = {"free": "free bias (current)",
              "gated": "gated: 0 when input off",
              "anchored": r"anchored: $NN(x)-NN(x_{off})$"}
    for j, mode in enumerate(modes):
        ax_m, ax_p = axes[0, j], axes[1, j]
        for n in names:
            dark, pale = BRANCH_COLORS.get(n, ("k", "0.7"))
            t = truth[f"truth_{n}"]
            ax_m.plot(t.mean(0) / scale, p, color=pale, lw=5, zorder=1)
            ax_p.plot(np.abs(t - t.mean(0)).mean(0) / scale, p, color=pale,
                      lw=5, zorder=1)
            for r in runs[mode]:
                f = r[f"branch_{n}"]
                ax_m.plot(f.mean(0) / scale, p, color=dark, lw=1.0, zorder=3)
                ax_p.plot(np.abs(f - f.mean(0)).mean(0) / scale, p, color=dark,
                          lw=1.0, zorder=3)
        for r in runs[mode]:
            ax_m.plot(r["intercept"] / scale, p, color="0.3", lw=1.0, ls="--",
                      zorder=2)
        ax_m.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax_m.set_title(titles[mode], fontsize=10)
        ax_m.set_xlabel(r"Mean $E[f_b]$, Q bias [$10^{-9}$]")
        ax_p.set_xlabel(r"Mean $|f_b - E f_b|$ [$10^{-9}$]")
        ax_p.set_xlim(left=0)
        for ax in (ax_m, ax_p):
            ax.grid(alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Pressure (hPa)")
        ax.set_ylim(p.max(), p.min())

    from matplotlib.lines import Line2D
    handles = []
    for n in names:
        dark, pale = BRANCH_COLORS.get(n, ("k", "0.7"))
        handles += [Line2D([], [], color=pale, lw=5, label=f"{n} truth"),
                    Line2D([], [], color=dark, lw=1, label=f"{n} PANN (per seed)")]
    handles.append(Line2D([], [], color="0.3", lw=1, ls="--",
                          label="global intercept (truth 0)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.005, 0.74, "(a) mean\ncontribution", rotation=90, va="center",
             fontsize=9)
    fig.text(0.005, 0.30, "(b) variance\nShapley", rotation=90, va="center",
             fontsize=9)
    fig.tight_layout(rect=(0.02, 0.06, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(REPO / "config" / "config.yaml"))
    ap.add_argument("--data", default=str(REPO / "data" /
                                          "training_dataset_partitioned.nc"))
    ap.add_argument("--out_dir", default=str(REPO / "output" / "experiments" /
                                             "zero_anchored_bias"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg["target"]["mode"] != "simple_gaussian_q":
        raise ValueError("this experiment needs target.mode: simple_gaussian_q "
                         "(planted truth)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        run_all(args, cfg)

    truth, names, runs = load_runs(out_dir)
    rows = summarise(truth, names, runs)
    print_table(rows, names)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1))
    plot(truth, names, runs, out_dir / "biasMode_comparison.png")


if __name__ == "__main__":
    main()
