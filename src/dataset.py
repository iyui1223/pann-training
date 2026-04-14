"""
Dataset loading and normalisation for PANN training.

Expects M03 ``training_dataset_partitioned.nc`` (step4b) with:

- Tendency blocks: ``{T,Q,QL,QI}_tendency_{conv,cloud,vdif,nogw}``
- PRAD heating rates: ``PHRSW``, ``PHRSC``, ``PHRLW``, ``PHRLC``

Target modes
------------
- ``scm_minus_reference`` (default): T and Q bias (SCM total - actual).
- ``simple_linear_q``: Q-only synthetic bias from conv and cloud branches
  (linear combination for architecture validation).
- ``simple_gaussian_q``: Q-only synthetic bias with Gaussian height-dependent
  envelopes (non-linear validation of vertical structure learning).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

TENDENCY_PREFIXES = ("T", "Q", "QL", "QI")
INITIAL_STATE_VARS = tuple(f"{v}_initial" for v in TENDENCY_PREFIXES)

PRAD_RATE_VARS = ("PHRSW", "PHRSC", "PHRLW", "PHRLC")
PRAD_BRANCH_NAMES = ("rad_sw", "rad_sw_clear", "rad_lw", "rad_lw_clear")

TENDENCY_SUFFIXES_PARTITION = ("_conv", "_cloud", "_vdif", "_nogw")

BIAS_VARS = ("T", "Q")

PARTITION_BRANCH_NAMES = (
    "conv",
    "cloud",
) + PRAD_BRANCH_NAMES + ("vdif", "nogw")

TARGET_MODE_SCM_MINUS_REFERENCE = "scm_minus_reference"
TARGET_MODE_SIMPLE_LINEAR_Q = "simple_linear_q"
TARGET_MODE_SIMPLE_GAUSSIAN_Q = "simple_gaussian_q"
TARGET_MODE_CEDA_DQ = "ceda_dq"

BRANCH_LABELS = {
    "conv": "Convection",
    "cloud": "Cloud / large-scale",
    "rad_sw": "Radiation SW (all-sky)",
    "rad_sw_clear": "Radiation SW (clear-sky)",
    "rad_lw": "Radiation LW (all-sky)",
    "rad_lw_clear": "Radiation LW (clear-sky)",
    "vdif": "VDF + orographic GWD",
    "nogw": "Non-orographic GWD",
}


def branch_label(name):
    return BRANCH_LABELS.get(name, name)


def _stack_variables(ds, var_names):
    arrays = [ds[v].values for v in var_names]
    stacked = np.stack(arrays, axis=1)
    return stacked.reshape(stacked.shape[0], -1)


def _tendency_var_names(process_suffix):
    return [f"{v}_tendency{process_suffix}" for v in TENDENCY_PREFIXES]


def _missing_partition_fields(ds):
    """Return list of required NetCDF variables that are absent."""
    missing = []
    for v in PRAD_RATE_VARS:
        if v not in ds.data_vars:
            missing.append(v)
    for suf in TENDENCY_SUFFIXES_PARTITION:
        for vn in _tendency_var_names(suf):
            if vn not in ds.data_vars:
                missing.append(vn)
    return missing


def compute_blocks(n_levels, block_size, overlap):
    """Return list of (start, end) tuples for overlapping level blocks.

    Adjacent blocks share ``overlap`` levels.  The last block is extended
    backward so every block has exactly ``block_size`` levels.
    """
    stride = block_size - overlap
    blocks = []
    start = 0
    while start < n_levels:
        end = min(start + block_size, n_levels)
        if end - start < block_size and blocks:
            start = max(n_levels - block_size, 0)
            end = n_levels
        blocks.append((start, end))
        if end >= n_levels:
            break
        start += stride
    return blocks


def _normalise_target_cfg(target_cfg):
    if target_cfg is None:
        return {"mode": TARGET_MODE_SCM_MINUS_REFERENCE}
    if isinstance(target_cfg, str):
        return {"mode": target_cfg}
    return dict(target_cfg)


def _slice_raw_to_block(raw, block_idx, block_size, block_overlap):
    """Slice a full-profile raw dict to a single block's levels."""
    blocks = compute_blocks(raw["n_levels"], block_size, block_overlap)
    if block_idx < 0 or block_idx >= len(blocks):
        raise ValueError(
            f"block_idx={block_idx} out of range for {len(blocks)} blocks"
        )
    bstart, bend = blocks[block_idx]
    blk_len = bend - bstart
    n_out = raw["n_output_vars"]

    sliced_xs = []
    for x in raw["x_branch_list"]:
        n_vars = x.shape[1] // raw["n_levels"]
        parts = [x[:, v * raw["n_levels"] + bstart : v * raw["n_levels"] + bend]
                 for v in range(n_vars)]
        sliced_xs.append(np.concatenate(parts, axis=1))

    y_full = raw["y"]
    y_parts = [y_full[:, v * raw["n_levels"] + bstart : v * raw["n_levels"] + bend]
               for v in range(n_out)]
    sliced_y = np.concatenate(y_parts, axis=1)

    sliced = dict(raw)
    sliced["x_branch_list"] = sliced_xs
    sliced["y"] = sliced_y
    sliced["n_levels"] = blk_len
    sliced["pressure_pa"] = raw["pressure_pa"][bstart:bend]
    sliced["block_start"] = bstart
    sliced["block_end"] = bend
    sliced["full_n_levels"] = raw["n_levels"]
    sliced["n_input_vars"] = {
        name: sliced_xs[i].shape[1] // blk_len
        for i, name in enumerate(raw["branch_names"])
    }

    if "truth_envelopes" in raw:
        sliced["truth_envelopes"] = {
            k: v[bstart:bend] for k, v in raw["truth_envelopes"].items()
        }
    if "truth_weights" in raw:
        sliced["truth_weights"] = raw["truth_weights"]

    return sliced


class Scaler:
    """Per-feature zero-mean / unit-variance standardisation."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-30] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.std = d["std"]


class PANNDataset(Dataset):
    """Normalised branch inputs and target."""

    def __init__(self, xs_list, y):
        self.xs = [torch.as_tensor(x, dtype=torch.float32) for x in xs_list]
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.xs) + (self.y[idx],)


def load_raw_arrays(nc_path, n_levels=None, level_start=None, target_cfg=None,
                    block_idx=None, block_size=None, block_overlap=None):
    """Load partitioned NetCDF into branch arrays and bias target ``y``.

    Parameters
    ----------
    n_levels : int or None
        Keep only the first ``n_levels`` levels (from TOA).
    level_start : int or None
        Skip the first ``level_start`` levels (0-based).
    target_cfg : dict or None
        Training config key ``target``.  ``mode`` selects the target.
    block_idx : int or None
        When set, slice all arrays to the specified block after loading.
        Requires ``block_size`` and ``block_overlap``.
    block_size, block_overlap : int or None
        Level-blocking parameters.  See :func:`compute_blocks`.
    """
    ds = xr.open_dataset(nc_path)
    lev_lo = level_start or 0
    lev_hi = n_levels  # None → all
    ds = ds.isel(level=slice(lev_lo, lev_hi))
    n_lev = ds.sizes["level"]

    tc = _normalise_target_cfg(target_cfg)
    mode = tc.get("mode", TARGET_MODE_SCM_MINUS_REFERENCE)

    pressure_pa = np.asarray(
        ds["pressure_initial"].values, dtype=np.float64,
    ).mean(axis=0).ravel()[:n_lev]

    # ── simple_linear_q: 2 Q-only branches, synthetic linear target ──
    if mode == TARGET_MODE_SIMPLE_LINEAR_Q:
        sub = tc.get("simple_linear_q") or {}
        conv_scale = float(sub.get("conv_scale", 0.2))
        cloud_scale = float(sub.get("cloud_scale", 2.0))

        x_conv_q = np.asarray(
            ds["Q_tendency_conv"].values, dtype=np.float64,
        ).reshape(-1, n_lev)
        x_cloud_q = np.asarray(
            ds["Q_tendency_cloud"].values, dtype=np.float64,
        ).reshape(-1, n_lev)

        y = conv_scale * x_conv_q + cloud_scale * x_cloud_q

        ds.close()

        branch_names = ("conv", "cloud")
        xs = [x_conv_q, x_cloud_q]
        n_input_vars = {name: 1 for name in branch_names}

        result = {
            "branch_names": branch_names,
            "x_branch_list": xs,
            "y": y,
            "n_levels": n_lev,
            "n_samples": y.shape[0],
            "n_input_vars": n_input_vars,
            "n_output_vars": 1,
            "target_mode": mode,
            "pressure_pa": pressure_pa,
            "truth_weights": {"conv": conv_scale, "cloud": cloud_scale},
        }
        if block_idx is not None:
            result = _slice_raw_to_block(result, block_idx, block_size, block_overlap)
        return result

    # ── simple_gaussian_q: 2 Q-only branches, Gaussian envelopes ───
    if mode == TARGET_MODE_SIMPLE_GAUSSIAN_Q:
        sub = tc.get("simple_gaussian_q") or {}
        p_hpa = pressure_pa / 100.0

        x_conv_q = np.asarray(
            ds["Q_tendency_conv"].values, dtype=np.float64,
        ).reshape(-1, n_lev)
        x_cloud_q = np.asarray(
            ds["Q_tendency_cloud"].values, dtype=np.float64,
        ).reshape(-1, n_lev)

        def _gaussian_envelope(center_hpa, sigma_levels):
            k_center = int(np.argmin(np.abs(p_hpa - center_hpa)))
            k = np.arange(n_lev)
            return np.exp(-0.5 * ((k - k_center) / sigma_levels) ** 2)

        conv_scale = float(sub.get("conv_scale", 1.1))
        conv_center = float(sub.get("conv_center_hpa", 200))
        conv_sigma = float(sub.get("conv_sigma_levels", 2))
        cloud_scale = float(sub.get("cloud_scale", 0.9))
        cloud_center = float(sub.get("cloud_center_hpa", 500))
        cloud_sigma = float(sub.get("cloud_sigma_levels", 5))

        env_conv = _gaussian_envelope(conv_center, conv_sigma)
        env_cloud = _gaussian_envelope(cloud_center, cloud_sigma)

        y = (conv_scale * env_conv[None, :] * x_conv_q
             + cloud_scale * env_cloud[None, :] * x_cloud_q)

        ds.close()

        branch_names = ("conv", "cloud")
        xs = [x_conv_q, x_cloud_q]
        n_input_vars = {name: 1 for name in branch_names}

        result = {
            "branch_names": branch_names,
            "x_branch_list": xs,
            "y": y,
            "n_levels": n_lev,
            "n_samples": y.shape[0],
            "n_input_vars": n_input_vars,
            "n_output_vars": 1,
            "target_mode": mode,
            "pressure_pa": pressure_pa,
            "truth_envelopes": {
                "conv": conv_scale * env_conv,
                "cloud": cloud_scale * env_cloud,
            },
        }
        if block_idx is not None:
            result = _slice_raw_to_block(result, block_idx, block_size, block_overlap)
        return result

    # ── scm_minus_reference: full 8-branch, T+Q bias (original) ──────
    if mode != TARGET_MODE_SCM_MINUS_REFERENCE:
        ds.close()
        raise ValueError("unknown target.mode %r" % (mode,))

    missing = _missing_partition_fields(ds)
    if missing:
        ds.close()
        msg = ", ".join(missing[:25])
        extra = f" (+{len(missing) - 25} more)" if len(missing) > 25 else ""
        raise ValueError(
            "NetCDF is missing required partitioned fields (use M03 step4b). "
            f"Missing: {msg}{extra}"
        )

    initial_state = _stack_variables(ds, INITIAL_STATE_VARS)
    x_conv = _stack_variables(ds, _tendency_var_names("_conv"))
    x_cloud = _stack_variables(ds, _tendency_var_names("_cloud"))

    bias_parts = []
    for v in BIAS_VARS:
        scm = ds[f"{v}_tendency_total"].values
        ref = ds[f"{v}_tendency_actual"].values
        bias_parts.append(scm - ref)
    y = np.concatenate(bias_parts, axis=1)

    xs = [x_conv, x_cloud]
    for nc_name in PRAD_RATE_VARS:
        xs.append(
            np.asarray(ds[nc_name].values, dtype=np.float64).reshape(-1, n_lev)
        )
    tend_vdif = _stack_variables(ds, _tendency_var_names("_vdif"))
    tend_nogw = _stack_variables(ds, _tendency_var_names("_nogw"))
    xs.append(np.concatenate([tend_vdif, initial_state], axis=1))
    xs.append(np.concatenate([tend_nogw, initial_state], axis=1))

    ds.close()

    n_input_vars = {
        name: xs[i].shape[1] // n_lev
        for i, name in enumerate(PARTITION_BRANCH_NAMES)
    }

    result = {
        "branch_names": PARTITION_BRANCH_NAMES,
        "x_branch_list": xs,
        "y": y,
        "n_levels": n_lev,
        "n_samples": y.shape[0],
        "n_input_vars": n_input_vars,
        "n_output_vars": 2,
        "target_mode": mode,
        "pressure_pa": pressure_pa,
    }
    if block_idx is not None:
        result = _slice_raw_to_block(result, block_idx, block_size, block_overlap)
    return result


def kfold_indices(n_samples, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[: n_samples % n_folds] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        val_idx = indices[start : start + size]
        train_idx = np.concatenate([indices[:start], indices[start + size :]])
        folds.append((train_idx, val_idx))
        start += size
    return folds


def make_dataloaders(raw, train_idx, val_idx, batch_size=64):
    names = raw["branch_names"]
    xs = raw["x_branch_list"]

    scalers = {name: Scaler().fit(xs[i][train_idx]) for i, name in enumerate(names)}
    scaler_y = Scaler().fit(raw["y"][train_idx])

    xs_n = [
        scalers[name].transform(xs[i]).astype(np.float32)
        for i, name in enumerate(names)
    ]
    y_n = scaler_y.transform(raw["y"]).astype(np.float32)

    train_xs = [x[train_idx] for x in xs_n]
    val_xs = [x[val_idx] for x in xs_n]

    train_ds = PANNDataset(train_xs, y_n[train_idx])
    val_ds = PANNDataset(val_xs, y_n[val_idx])

    scalers_out = {**scalers, "y": scaler_y}

    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "scalers": scalers_out,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def save_branch_scalers(scalers, branch_names, path):
    payload = {f"{name}_mean": scalers[name].mean for name in branch_names}
    payload.update({f"{name}_std": scalers[name].std for name in branch_names})
    payload["y_mean"] = scalers["y"].mean
    payload["y_std"] = scalers["y"].std
    np.savez(path, **payload)


def load_branch_scalers(path, branch_names):
    raw = np.load(path)
    scalers = {}
    for name in branch_names:
        s = Scaler()
        s.load_state_dict({
            "mean": raw[f"{name}_mean"],
            "std": raw[f"{name}_std"],
        })
        scalers[name] = s
    s_y = Scaler()
    s_y.load_state_dict({"mean": raw["y_mean"], "std": raw["y_std"]})
    scalers["y"] = s_y
    return scalers
