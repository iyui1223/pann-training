"""Branch-bias identifiability with an intermittent (sometimes-off) predictor.

Toy counterpart of the PANN mean-contribution problem, in the style of
``correlated_nonpredictor_shapley.py``.  Two correlated inputs:

  * ``x1`` -- "convection": exactly 0 (process off) a fraction
              ``1 - ACTIVE_FRAC`` of the time, as for the PANN conv branch.
  * ``x2`` -- "cloud": always on.

Planted truth is the first-order (small-dt) error: linear in each process
tendency, plus an optional per-process offset that exists only while the
process is on (e.g. a convective trigger):

    f_b = 1[x_b != 0] * (b_b * x_b + c_b),      y = f_1 + f_2 + noise

With ``c_b = 0`` the error vanishes with the input; with ``c_b != 0`` it
jumps at zero.  True mean contributions are
E[f_b] = b_b * E[x_b] + c_b * P(x_b != 0)  (exactly 0 for an off branch).
Since x2 is always on, ``c_2`` is indistinguishable from a global constant;
the interesting offset is ``c_1``.

As in the PANN, inputs and target are z-scored before fitting and every
model carries one shared, unattributed global intercept ``c0``.  Each
branch is LINEAR (w_b * xn_b plus a bias), so three bias treatments are:

  free      f_b = w_b*xn_b + c_b                current PANN: bias always on
  gated     f_b = g_b * (w_b*xn_b + c_b)        g_b = 1[x_b != 0]: output,
                                                bias included, is switched
                                                off when the process is off
  anchored  f_b = w_b*(xn_b - xn_off_b)         = NN(x) - NN(x_off); xn_off is
                                                the z-scored physical zero

Fitting is exact least squares.  When the design is rank-deficient (biases
that only enter as a sum), plain gradient descent from an initial point w0
converges to  pinv(A) y + (I - pinv(A) A) w0 : the identified part is fixed
by the data, the null-space part stays at its random init.  We use that
closed form with a random w0 per seed -- this mimics NN training (random
bias init, tiny weight decay), and the seed spread of E[f_b] is a direct
measure of how un-identified the branch biases are.

Per branch we report
  E[f_b]          mean contribution  (bias-sensitive; the hard plot)
  E|f_b - E f_b|  variance Shapley   (bias-free; the easy plot)

Run:  python experiments/intermittent_branch_bias.py
"""

import numpy as np

# ----------------------------- CONFIG (edit me) ------------------------------
SEED = 0
N_SAMPLES = 100_000
NOISE_STD = 0.1
ACTIVE_FRAC = 0.7       # fraction of samples where x1 ("convection") is on
N_INIT_SEEDS = 5        # random inits of the un-identified (null-space) part
INIT_STD = 0.5          # std of the random init in normalised units

# x_b = offset_b + z_b with corr(z1, z2) = rho; x1 is then zeroed when off.
# "biases" (optional, default 0) are the on-only offsets c_b.
SCENARIOS = [
    {
        "name": "A. two correlated true predictors, x1 off 30% (rho = 0.7)",
        "rho": 0.7,
        "offsets": [1.0, 1.0],
        "betas": [1.0, 0.5],
    },
    {
        "name": ("B. intermittent x1 is a pure correlate (beta1 = 0), "
                 "x2 the only predictor (rho = 0.9)"),
        "rho": 0.9,
        "offsets": [1.0, 1.0],
        "betas": [0.0, 1.0],
    },
    {
        "name": ("C. as A, plus an on-only offset in x1 "
                 "(trigger jump c1 = 0.5)"),
        "rho": 0.7,
        "offsets": [1.0, 1.0],
        "betas": [1.0, 0.5],
        "biases": [0.5, 0.0],
    },
]
# -----------------------------------------------------------------------------

BRANCHES = ("x1", "x2")
VARIANTS = ("free", "gated", "anchored")
# Exactly-duplicated bias columns leave ~1e-13 singular values from round-off.
RCOND = 1e-10


def make_data(rng, scenario):
    rho = scenario["rho"]
    cov = [[1.0, rho], [rho, 1.0]]
    z = rng.multivariate_normal(np.zeros(2), cov, size=N_SAMPLES)
    x = z + np.asarray(scenario["offsets"], dtype=float)
    active1 = rng.random(N_SAMPLES) < ACTIVE_FRAC
    x[~active1, 0] = 0.0
    biases = np.asarray(scenario.get("biases", [0.0, 0.0]), dtype=float)
    contrib = (x != 0) * (x * np.asarray(scenario["betas"], dtype=float) + biases)
    y = contrib.sum(axis=1) + rng.normal(0.0, NOISE_STD, N_SAMPLES)
    return x, y, contrib


def zscore(v):
    mu, sd = v.mean(axis=0), v.std(axis=0)
    return (v - mu) / sd, mu, sd


def design(variant, x):
    """Columns of the design matrix, each tagged with its owner.

    Owner is a branch name or ``"c0"`` (the unattributed global intercept).
    """
    xn, mu, sd = zscore(x)
    xn_off = -mu / sd
    one = np.ones(len(x))
    cols = []
    for j, b in enumerate(BRANCHES):
        if variant == "free":
            cols += [(b, xn[:, j]), (b, one)]
        elif variant == "gated":
            g = (x[:, j] != 0.0).astype(float)
            cols += [(b, g * xn[:, j]), (b, g)]
        elif variant == "anchored":
            cols += [(b, xn[:, j] - xn_off[j])]
    cols.append(("c0", one))
    owners = [o for o, _ in cols]
    return owners, np.column_stack([c for _, c in cols])


def fit(A, yn, w0):
    """Least squares; null-space component kept at the init ``w0``."""
    pinv = np.linalg.pinv(A, rcond=RCOND)
    return pinv @ yn + (np.eye(A.shape[1]) - pinv @ A) @ w0


def run_variant(variant, x, y, rng):
    owners, A = design(variant, x)
    yn, y_mu, y_sd = zscore(y)
    s = np.linalg.svd(A, compute_uv=False)
    null_dim = int(np.sum(s <= RCOND * s[0]))
    results = []
    for _ in range(N_INIT_SEEDS):
        coef = fit(A, yn, rng.normal(0.0, INIT_STD, A.shape[1]))
        res = {}
        for b in BRANCHES + ("c0",):
            mask = np.array([o == b for o in owners])
            res[b] = A[:, mask] @ coef[mask] * y_sd
        res["c0"] = res["c0"] + y_mu
        pred = sum(res.values())
        res["r2"] = 1.0 - np.var(y - pred) / np.var(y)
        results.append(res)
    return null_dim, results


def _ms(values):
    v = np.asarray(values)
    return f"{v.mean():+8.3f}+-{v.std():5.3f}"


def report(scenario, x, contrib, runs):
    active = (x[:, 0] != 0).mean()
    corr = np.corrcoef(x, rowvar=False)[0, 1]
    b1, b2 = scenario["betas"]
    c1, c2 = scenario.get("biases", [0.0, 0.0])
    print("\n" + "=" * 96)
    print(scenario["name"])
    print("=" * 96)
    print(f"y = 1[x1!=0]*({b1}*x1 + {c1}) + 1[x2!=0]*({b2}*x2 + {c2}) + noise"
          f"   x1 active {active:.0%}   "
          f"realised corr(x1, x2) = {corr:.3f}")
    print("Values: mean +- std over random inits (std > 0 => un-identified)\n")
    hdr = (f"  {'variant':<10}{'null':>5}{'R2':>8}"
           f"{'E[f_x1]':>17}{'E[f_x2]':>17}{'c0':>17}"
           f"{'E|phi_x1|':>17}{'E|phi_x2|':>17}")
    print(hdr)

    t_mean = contrib.mean(axis=0)
    t_phi = np.abs(contrib - t_mean).mean(axis=0)
    print(f"  {'truth':<10}{'':>5}{'':>8}"
          f"{t_mean[0]:>+11.3f}{'':6}{t_mean[1]:>+11.3f}{'':6}"
          f"{0.0:>+11.3f}{'':6}{t_phi[0]:>11.3f}{'':6}{t_phi[1]:>11.3f}")

    for variant, (null_dim, res) in runs.items():
        means = {b: [r[b].mean() for r in res] for b in BRANCHES}
        phis = {b: [np.abs(r[b] - r[b].mean()).mean() for r in res]
                for b in BRANCHES}
        c0 = [r["c0"].mean() for r in res]
        r2 = np.mean([r["r2"] for r in res])
        print(f"  {variant:<10}{null_dim:>5}{r2:>8.3f}"
              f"{_ms(means['x1']):>17}{_ms(means['x2']):>17}{_ms(c0):>17}"
              f"{_ms(phis['x1']):>17}{_ms(phis['x2']):>17}")


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 96)
    print("Branch-bias identifiability: free vs gated vs anchored branch bias")
    print("=" * 96)
    print("null = rank deficiency of the design (# bias directions the data "
          "cannot pin down)")
    for scenario in SCENARIOS:
        x, y, contrib = make_data(rng, scenario)
        runs = {v: run_variant(v, x, y, rng) for v in VARIANTS}
        report(scenario, x, contrib, runs)


if __name__ == "__main__":
    main()
