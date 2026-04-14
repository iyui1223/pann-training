"""
Process-Additive Neural Network (PANN).

Independent branches (one per physics partition) whose outputs are summed
to predict the total SCM-vs-reference bias (per level: dT, dQ, ...).

Supports three branch architectures:
  - ``flat``:      constant-width MLP
  - ``hourglass``: encoder-bottleneck-decoder MLP with additive skip connections
  - ``conv1d``:    multi-scale 1-D convolution with bottleneck and symmetric
                   decoder — exploits vertical locality of tendency profiles
"""

import torch
import torch.nn as nn


# ── Branch architectures ────────────────────────────────────────────────────


def _halving_dims(start, stop):
    """Generate dimensions by halving: start, start//2, ..., stop."""
    dims = []
    d = start
    while d >= stop:
        dims.append(d)
        if d == stop:
            break
        d = max(d // 2, stop)
    return dims


class FlatBranch(nn.Module):
    """Constant-width MLP branch (original architecture)."""

    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_hidden_layers=4, dropout=0.0):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, save_hidden=False):
        hidden_activations = []
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))
            if save_hidden:
                hidden_activations.append(x.detach().cpu())
        x = self.output_layer(x)
        if save_hidden:
            return x, hidden_activations
        return x


class HourglassBranch(nn.Module):
    """Encoder-bottleneck-decoder MLP with additive skip connections."""

    def __init__(self, input_dim, hidden_dim, output_dim,
                 bottleneck_dim=64, dropout=0.0):
        super().__init__()
        enc_dims = _halving_dims(hidden_dim, bottleneck_dim)
        dec_dims = list(reversed(enc_dims[:-1]))

        self.encoder = nn.ModuleList()
        prev = input_dim
        for d in enc_dims:
            self.encoder.append(nn.Linear(prev, d))
            prev = d

        self.decoder = nn.ModuleList()
        for d in dec_dims:
            self.decoder.append(nn.Linear(prev, d))
            prev = d

        self.output_layer = nn.Linear(prev, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._n_skip = len(dec_dims)

    def forward(self, x, save_hidden=False):
        hidden_activations = []
        enc_acts = []

        for layer in self.encoder:
            x = self.dropout(self.activation(layer(x)))
            enc_acts.append(x)
            if save_hidden:
                hidden_activations.append(x.detach().cpu())

        skip_sources = list(reversed(enc_acts[:-1]))

        for i, layer in enumerate(self.decoder):
            x = self.dropout(self.activation(layer(x)))
            if i < len(skip_sources):
                x = x + skip_sources[i]
            if save_hidden:
                hidden_activations.append(x.detach().cpu())

        x = self.output_layer(x)
        if save_hidden:
            return x, hidden_activations
        return x


class ConvBranch(nn.Module):
    """Multi-scale 1-D conv branch with bottleneck and symmetric decoder.

    Exploits vertical locality: nearby levels share bias characteristics,
    so weight-sharing across levels is more efficient than FC for small
    sample sizes.

    Encoder: three parallel Conv1d paths (local k=3, mid-scale k=7,
    dilated k=3/d=4) -> concat -> 1x1 merge -> refinement (2x Conv k=3).
    Bottleneck: 1x1 conv compressing to ``bottleneck_dim`` channels
    ("vertical modes").
    Decoder: mirrors refinement with additive skip connections from encoder,
    then 1x1 projection to output channels.
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 bottleneck_dim=10, dropout=0.0,
                 conv_channels=None):
        super().__init__()
        C = conv_channels or hidden_dim
        n_output_vars = output_dim // input_dim if input_dim > 0 else 1
        self._n_output_vars = n_output_vars
        self._input_levels = input_dim

        act = nn.ReLU()
        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Multi-scale encoder paths (input: 1 channel)
        self.path_local = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), act, drop,
        )
        self.path_mid = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3), act, drop,
        )
        self.path_dilated = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, dilation=4, padding=4), act, drop,
        )

        # 1x1 merge (24 -> C)
        self.merge = nn.Sequential(
            nn.Conv1d(24, C, kernel_size=1), act, drop,
        )

        # Refinement (encoder side)
        self.refine1 = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1), act, drop,
        )
        self.refine2 = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1), act, drop,
        )

        # Bottleneck
        self.to_bottleneck = nn.Sequential(
            nn.Conv1d(C, bottleneck_dim, kernel_size=1), act, drop,
        )

        # Decoder (symmetric to refinement)
        self.from_bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_dim, C, kernel_size=3, padding=1), act, drop,
        )
        self.decode1 = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1), act, drop,
        )

        # Output projection
        self.output_proj = nn.Conv1d(C, n_output_vars, kernel_size=1)

    def forward(self, x, save_hidden=False):
        # x: (batch, input_dim) -> (batch, 1, L)
        x = x.unsqueeze(1)

        # Multi-scale encoder
        p1 = self.path_local(x)
        p2 = self.path_mid(x)
        p3 = self.path_dilated(x)
        x = torch.cat([p1, p2, p3], dim=1)  # (batch, 24, L)

        x = self.merge(x)  # (batch, C, L)

        # Refinement with saved activations for skip connections
        r1 = self.refine1(x)   # skip target for decode1
        r2 = self.refine2(r1)  # skip target for from_bottleneck

        # Bottleneck
        bn = self.to_bottleneck(r2)  # (batch, bottleneck_dim, L)

        hidden_activations = []
        if save_hidden:
            hidden_activations.append(bn.detach().cpu())

        # Decoder with additive skips
        d1 = self.from_bottleneck(bn) + r2
        d2 = self.decode1(d1) + r1

        out = self.output_proj(d2)  # (batch, n_output_vars, L)
        out = out.reshape(out.size(0), -1)  # (batch, output_dim)

        if save_hidden:
            return out, hidden_activations
        return out


def make_branch(architecture, input_dim, hidden_dim, output_dim, **kwargs):
    if architecture == "flat":
        return FlatBranch(
            input_dim, hidden_dim, output_dim,
            n_hidden_layers=kwargs.get("n_hidden_layers", 4),
            dropout=kwargs.get("dropout", 0.0),
        )
    if architecture == "hourglass":
        return HourglassBranch(
            input_dim, hidden_dim, output_dim,
            bottleneck_dim=kwargs.get("bottleneck_dim", 64),
            dropout=kwargs.get("dropout", 0.0),
        )
    if architecture == "conv1d":
        return ConvBranch(
            input_dim, hidden_dim, output_dim,
            bottleneck_dim=kwargs.get("bottleneck_dim", 10),
            dropout=kwargs.get("dropout", 0.0),
            conv_channels=kwargs.get("conv_channels"),
        )
    raise ValueError(f"Unknown architecture: {architecture!r}")


# ── PANN ────────────────────────────────────────────────────────────────────


class PANN(nn.Module):
    """Process-Additive Neural Network: one branch per partition, outputs summed.

    Parameters
    ----------
    n_levels : int
        Vertical levels (each branch outputs ``n_output_vars * n_levels``).
    hidden_dim : int
        Largest hidden-layer width (FC) or channel count after merge (conv1d).
    branches : list of (name, input_dim)
        Ordered branch specs; ``input_dim`` is the **flattened** input size.
    architecture : str
        ``"flat"``, ``"hourglass"``, or ``"conv1d"``.
    n_hidden_layers, bottleneck_dim, dropout, conv_channels
        Passed to branch factory.
    n_output_vars : int
        Typically 2 (dT and dQ bias per level).
    """

    def __init__(
        self,
        n_levels,
        hidden_dim,
        branches,
        architecture="flat",
        n_hidden_layers=4,
        n_output_vars=2,
        bottleneck_dim=64,
        dropout=0.0,
        conv_channels=None,
    ):
        super().__init__()
        if not branches:
            raise ValueError("PANN requires a non-empty ``branches`` list")
        self.n_levels = n_levels
        self.n_output_vars = n_output_vars
        output_dim = n_output_vars * n_levels

        self.branch_names = tuple(b[0] for b in branches)
        branch_dims = {b[0]: int(b[1]) for b in branches}

        self.branches = nn.ModuleDict(
            {
                name: make_branch(
                    architecture,
                    branch_dims[name],
                    hidden_dim,
                    output_dim,
                    n_hidden_layers=n_hidden_layers,
                    bottleneck_dim=bottleneck_dim,
                    dropout=dropout,
                    conv_channels=conv_channels,
                )
                for name in self.branch_names
            }
        )

        self.branch_mask = {name: True for name in self.branch_names}
        self.branch_weights = {name: 1.0 for name in self.branch_names}

    def set_branch_mask(self, mask_dict):
        """Permanently enable/disable branches.

        Parameters
        ----------
        mask_dict : dict[str, bool]
            Maps branch name to ``True`` (active) or ``False`` (dropped).
        """
        for name, active in mask_dict.items():
            if name in self.branch_mask:
                self.branch_mask[name] = active

    def set_branch_weights(self, weights_dict):
        """Set soft importance weights for active branches.

        Parameters
        ----------
        weights_dict : dict[str, float]
            Maps branch name to a positive weight.  Weights are expected to
            sum to ``N`` (number of branches) so that the total output
            magnitude is preserved on average.
        """
        for name, w in weights_dict.items():
            if name in self.branch_weights:
                self.branch_weights[name] = float(w)

    def forward(self, *xs, save_hidden=False, expose_all_branches=False):
        """Forward pass. Pass one tensor per branch in ``self.branch_names`` order.

        Branches with ``branch_mask[name] == False`` produce zero contribution
        to ``total``.

        If ``save_hidden`` and ``expose_all_branches``, every submodule still
        runs so diagnostics (e.g. hidden extraction) see true activations;
        ``total`` sums only active branch outputs.
        """
        if len(xs) != len(self.branch_names):
            raise ValueError(
                f"Expected {len(self.branch_names)} inputs ({self.branch_names}), "
                f"got {len(xs)}"
            )

        if save_hidden:
            branch_outputs = {}
            total = 0
            for name, x in zip(self.branch_names, xs):
                w = self.branch_weights.get(name, 1.0)
                if expose_all_branches:
                    out, h = self.branches[name](x, save_hidden=True)
                    branch_outputs[name] = (out, h)
                    if self.branch_mask.get(name, True):
                        total = total + w * out
                    continue
                if not self.branch_mask.get(name, True):
                    zero = torch.zeros(x.size(0), self.n_output_vars * self.n_levels,
                                       device=x.device, dtype=x.dtype)
                    branch_outputs[name] = (zero, [])
                    continue
                out, h = self.branches[name](x, save_hidden=True)
                branch_outputs[name] = (out, h)
                total = total + w * out
            return total, branch_outputs

        total = 0
        for name, x in zip(self.branch_names, xs):
            if not self.branch_mask.get(name, True):
                continue
            w = self.branch_weights.get(name, 1.0)
            total = total + w * self.branches[name](x)
        return total
