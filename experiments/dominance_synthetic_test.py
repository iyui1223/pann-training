"""Synthetic sanity test for dominance / Shapley feature-importance analysis.

Two unit-variance, correlated features predict ``y``.  The script compares
several notions of "feature importance" so you can see how each behaves when
the predictors are correlated:

  1. Naive structural-coefficient shares  -- what you might guess from the betas
     (both the |beta| share and the beta^2 share).  These ignore covariance.
  2. Analytic general dominance (Shapley R2)  -- computed in closed form from the
     population covariance matrix and the true betas.  This is the ground truth.
  3. Empirical general dominance (Shapley R2)  -- computed transparently with
     scikit-learn by fitting ordinary least squares over *all* feature subsets.
  4. The ``dominance-analysis`` PyPI package  -- run only if it is importable.

General dominance == the Shapley value of each predictor's contribution to the
model R2; the values sum to the full-model R2.

----------------------------------------------------------------------------
HOW TO EXPERIMENT
----------------------------------------------------------------------------
Edit the CONFIG block below:
  * ``CORR``  -- feature correlation matrix (unit variances, so corr == cov).
                 Change the off-diagonal to set how strongly features covary.
  * ``BETAS`` -- the true structural coefficients (the "importance" you plant).
  * ``NOISE_STD`` -- additive Gaussian noise on y (0 => full-model R2 == 1).
Then re-run and compare the planted importance with each estimate.

Note on the worked example in the task description:
  x2 = -0.7*x1 + 0.3*z does NOT give Var(x2)=1 / corr=-0.7; it gives
  Var(x2)=0.49+0.09=0.58 and corr=-0.919.  To obtain unit variance with a
  target correlation rho we use  x2 = rho*x1 + sqrt(1-rho^2)*z.  This script
  builds the features directly from ``CORR`` via a multivariate normal draw,
  which generalises cleanly to more than two features.
"""

import itertools
from math import factorial

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ----------------------------- CONFIG (edit me) ------------------------------
SEED = 0
N_SAMPLES = 100_000

FEATURE_NAMES = ["x1", "x2"]

# Feature correlation matrix (unit variances -> this is also the covariance).
# Off-diagonal = Pearson r between features.  Default: r(x1, x2) = -0.7.
CORR = [
    [1.0, -0.7],
    [-0.7, 1.0],
]

# True structural coefficients: y = CONST + sum(BETAS * X) + noise
BETAS = [0.7, 0.3]

CONST = 5.0
# A little noise keeps the analysis non-degenerate.  iid noise scales every
# subset's R2 by the same factor, so the relative dominance shares are
# unchanged, but the dominance-analysis package needs full-model R2 < 1
# (it divides by zero when R2 == 1 exactly).  Set to 0.0 to disable.
NOISE_STD = 0.3
# -----------------------------------------------------------------------------


def make_data(seed, n, corr, betas, const, noise_std):
    rng = np.random.default_rng(seed)
    corr = np.asarray(corr, dtype=float)
    betas = np.asarray(betas, dtype=float)
    k = corr.shape[0]
    X = rng.multivariate_normal(mean=np.zeros(k), cov=corr, size=n)
    noise = rng.normal(0.0, noise_std, size=n) if noise_std > 0 else 0.0
    y = const + X @ betas + noise
    return X, y


def shapley_from_r2(feature_idx, r2_func):
    """General dominance (Shapley R2) for each feature given an R2(subset) func.

    phi_i = sum over subsets S not containing i of
            [|S|! (k-|S|-1)! / k!] * (R2(S u {i}) - R2(S))
    The phi_i sum to R2(full set).
    """
    k = len(feature_idx)
    phi = {i: 0.0 for i in feature_idx}
    others = {i: [j for j in feature_idx if j != i] for i in feature_idx}
    for i in feature_idx:
        rest = others[i]
        for s in range(len(rest) + 1):
            weight = factorial(s) * factorial(k - s - 1) / factorial(k)
            for subset in itertools.combinations(rest, s):
                base = r2_func(subset)
                withi = r2_func(tuple(sorted(subset + (i,))))
                phi[i] += weight * (withi - base)
    return phi


def analytic_r2_func(corr, betas, noise_std):
    """Population R2 of regressing y on a subset of features (closed form)."""
    C = np.asarray(corr, dtype=float)
    b = np.asarray(betas, dtype=float)
    var_y = float(b @ C @ b + noise_std ** 2)

    def r2(subset):
        if not subset:
            return 0.0
        ix = list(subset)
        Css = C[np.ix_(ix, ix)]
        cs = C[ix, :] @ b  # cov(x_subset, y)
        explained = float(cs @ np.linalg.solve(Css, cs))
        return explained / var_y

    return r2


def empirical_r2_func(X, y):
    """In-sample OLS R2 of regressing y on a subset of feature columns."""
    cache = {}

    def r2(subset):
        if not subset:
            return 0.0
        key = tuple(sorted(subset))
        if key not in cache:
            model = LinearRegression().fit(X[:, list(key)], y)
            cache[key] = float(model.score(X[:, list(key)], y))
        return cache[key]

    return r2


def as_shares(values):
    total = sum(values.values())
    if total == 0:
        return {k: float("nan") for k in values}
    return {k: v / total for k, v in values.items()}


def print_table(title, names, value_map, share_map):
    print(f"\n{title}")
    print(f"  {'feature':<10}{'value':>14}{'share %':>12}")
    for n in names:
        v = value_map[n]
        s = share_map[n] * 100 if share_map[n] == share_map[n] else float("nan")
        print(f"  {n:<10}{v:>14.4f}{s:>12.2f}")


def run_package(df, target, feature_names):
    """Run the dominance-analysis PyPI package if available."""
    try:
        from dominance_analysis import Dominance
    except Exception as exc:  # noqa: BLE001 - want any import failure reported
        print("\n[dominance-analysis package] NOT run:")
        print(f"  import failed: {exc!r}")
        print("  Install with:  pip install dominance-analysis")
        return

    print("\n[dominance-analysis package]")
    try:
        dom = Dominance(data=df, target=target, objective=1)
        # incremental_rsquare() == each feature's general (total) dominance,
        # i.e. its Shapley contribution to R2.  This is the number we want.
        incr = dom.incremental_rsquare()  # dict: feature -> general dominance R2
    except Exception as exc:  # noqa: BLE001
        print(f"  incremental_rsquare() FAILED: {exc!r}")
        return

    print("  incremental_rsquare()  (== general dominance / Shapley R2):")
    print(f"    {'feature':<10}{'value':>14}{'share %':>12}")
    total = sum(incr.values())
    for n in feature_names:
        val = incr.get(n, float("nan"))
        share = (val / total * 100) if total else float("nan")
        print(f"    {n:<10}{val:>14.4f}{share:>12.2f}")

    # dominance_stats() averages "partial dominance" over several model sizes
    # and is known to break for very few predictors (e.g. 2): it divides by a
    # zero Total-Dominance sum.  Guard it so the useful output above survives.
    try:
        stats = dom.dominance_stats()
        print("\n  dominance_stats():")
        with pd.option_context("display.width", 120,
                               "display.max_columns", None):
            print(stats.to_string())
    except Exception as exc:  # noqa: BLE001
        print(f"\n  dominance_stats() unavailable (expected with <3 features): "
              f"{exc!r}")


def main():
    names = list(FEATURE_NAMES)
    idx = list(range(len(names)))
    name_of = {i: names[i] for i in idx}

    X, y = make_data(SEED, N_SAMPLES, CORR, BETAS, CONST, NOISE_STD)

    # --- realised data summary -------------------------------------------------
    realised_corr = np.corrcoef(X, rowvar=False)
    print("=" * 68)
    print("Synthetic dominance-analysis test")
    print("=" * 68)
    print(f"n_samples={N_SAMPLES}  features={names}  betas={BETAS}  "
          f"const={CONST}  noise_std={NOISE_STD}")
    print("Target feature correlation matrix:")
    print(np.array(CORR))
    print("Realised feature correlation matrix:")
    print(np.round(realised_corr, 4))
    print(f"Realised feature variances: "
          f"{np.round(X.var(axis=0), 4).tolist()}")

    # --- 1. naive structural-coefficient shares --------------------------------
    abs_beta = {name_of[i]: abs(BETAS[i]) for i in idx}
    sq_beta = {name_of[i]: BETAS[i] ** 2 for i in idx}
    print_table("1. Naive |beta| share  (ignores covariance)",
                names, abs_beta, as_shares(abs_beta))
    print_table("1b. Naive beta^2 share (ignores covariance)",
                names, sq_beta, as_shares(sq_beta))

    # --- 2. analytic general dominance (population, ground truth) ---------------
    r2_pop = analytic_r2_func(CORR, BETAS, NOISE_STD)
    phi_pop = {name_of[i]: v for i, v in shapley_from_r2(idx, r2_pop).items()}
    print(f"\n(Analytic full-model R2 = {r2_pop(tuple(idx)):.4f})")
    print_table("2. Analytic general dominance (Shapley R2, population)",
                names, phi_pop, as_shares(phi_pop))

    # --- 3. empirical general dominance (sklearn over subsets) -----------------
    r2_emp = empirical_r2_func(X, y)
    phi_emp = {name_of[i]: v for i, v in shapley_from_r2(idx, r2_emp).items()}
    print(f"\n(Empirical full-model R2 = {r2_emp(tuple(idx)):.4f})")
    print_table("3. Empirical general dominance (Shapley R2, sklearn)",
                names, phi_emp, as_shares(phi_emp))

    # --- 4. dominance-analysis package -----------------------------------------
    df = pd.DataFrame(X, columns=names)
    df["y"] = y
    run_package(df, "y", names)

    print("\n" + "=" * 68)
    print("Interpretation: with negatively-correlated features the dominance")
    print("(Shapley) shares can differ markedly from the naive beta shares.")
    print("Edit CORR / BETAS at the top of the file and re-run to explore.")
    print("=" * 68)


if __name__ == "__main__":
    main()
