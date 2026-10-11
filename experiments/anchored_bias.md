# Anchored and gated branch biases in the PANN

Design note for the bias experiments in this directory:

- `intermittent_branch_bias.py` — numpy toy, closed-form fits (seconds).
- `zero_anchored_bias.py` (+ `_slurm.sh`) — same idea on the real
  `simple_gaussian_q` PANN target.

## 1. The problem: branch constants are not identifiable

The PANN predicts `y = c + Σ_b f_b(x_b)` and only the sum enters the loss.
For any constants `k_b` with `Σ k_b = 0` (or absorbed by `c`),

    f_b(x_b) → f_b(x_b) + k_b

leaves every prediction unchanged. Hence

| quantity | depends on the free constants? | figure |
|---|---|---|
| `E[f_b]` (mean contribution) | **yes** — arbitrary, set by init / optimiser | `branch_contributions_shapley.png` |
| `E\|f_b − E f_b\|` (variance Shapley) | no — constants cancel | `branch_shapley_profiles.png` |

This is why the variance plot matches the planted truth and the mean plot
does not. Two implementation details make it worse in the current code:
inputs are z-scored, so a physically "off" process is **not** zero inside the
network (`xn_off = −mean/std`); and `evaluate.py` adds `y_mean` to **every**
branch when converting to physical units.

## 2. Three branch-bias treatments

All three share one unattributed global intercept `c`.

| variant | branch output | `f_b` when process off |
|---|---|---|
| free (current) | `NN_b(x)` | `NN_b(x_off)`, arbitrary |
| gated | `1[x_b ≠ 0] · NN_b(x)` | 0 |
| anchored | `NN_b(x) − NN_b(x_off)` | 0 |

`x_off` is the normalised image of a physical zero input. Code:
`BiasModePANN` in `zero_anchored_bias.py`, `design()` in
`intermittent_branch_bias.py`.

### Key properties

1. **Anchored = free, reparameterised.** Any free model can be rewritten
   exactly as an anchored one:

       Σ_b NN_b(x_b) + c  =  Σ_b [NN_b(x_b) − NN_b(x_off)]  +  [c + Σ_b NN_b(x_off)]

   Same function class, same predictions, same R². Anchoring therefore does
   not need retraining: it can be applied **post hoc** to existing trained
   models by evaluating `f_b(x) − f_b(x_off)` and moving `Σ_b f_b(x_off)` into
   the intercept. It only fixes the bookkeeping, which is exactly what the
   mean plot needs.
2. **Gated is a different model class.** It adds a jump at `x_b = 0`
   (a trigger offset `c_b · 1[x_b ≠ 0]`), which neither free nor anchored can
   represent exactly. It identifies the constant of an intermittent branch
   (the "off" samples pin it down), but **not** of an always-on branch, whose
   constant stays confounded with `c`.
3. **What decides between them** is whether the true error jumps when a process
   switches on (toy scenario C): no jump → anchored is right; jump → gated.
4. **Diagnostic.** For the planted synthetic truth the true intercept is 0, so
   the learned `c` (in physical units) should go to ~0 when the decomposition
   is right.

## 3. Why linear-with-no-constant is the natural first-order truth

Write the step error as a function of the process increment
`Δq = dt · x(dt, s0)` and the initial state `s0`. No increment means no error,
so `e(0, s0) = 0`, and for small `dt`

    e / dt = J(s0) · x0(s0) + O(dt)

That is linear in the tendency, with **no constant**, and with a state-dependent
coefficient. The process errors also add at this order; operator-splitting
interactions between processes first appear at `O(dt²)`. A constant reappears
only if `e` jumps at `Δq = 0` (triggering), which gives the gated form
`c(s0) · 1[x ≠ 0] + J(s0) · x`.

## 4. Related work

**Anchored decompositions (the name and the maths).** Anchoring each component
at a reference point is the *anchored ANOVA* / *cut-HDMR* decomposition. The
usual *ANOVA* decomposition instead centres each component to zero mean.
Both are special cases of one projection framework.
- Rabitz & Aliş (1999), *General foundations of high-dimensional model
  representations*, J. Math. Chem. 25, 197–233.
  [doi:10.1023/A:1019188517934](https://doi.org/10.1023/A:1019188517934)
- Kuo, Sloan, Wasilkowski & Woźniakowski (2010), *On decompositions of
  multivariate functions*, Math. Comp. 79, 953–966.
  [doi:10.1090/S0025-5718-09-02319-9](https://doi.org/10.1090/S0025-5718-09-02319-9),
  [preprint](https://web.maths.unsw.edu.au/~fkuo/pubs/preprint/ksww09-decomp.pdf)
- Gao & Hesthaven (2011), *On ANOVA expansions and strategies for choosing the
  anchor point*, Appl. Math. Comput.
  [link](https://www.sciencedirect.com/science/article/abs/pii/S0096300310009252)
  (here the anchor is not a free choice: physics fixes it at "process off").

**Baseline attributions (same idea, applied after training).** For an additive
model, Baseline Shapley with baseline `x_off` gives exactly
`φ_b = f_b(x_b) − f_b(x_off)`, the anchored branch output. The current
evaluation, `φ_b = f_b − E[f_b]`, is the mean-baseline version: the constant
cancels, but the mean is then zero by construction.
- Sundararajan & Najmi (2020), *The Many Shapley Values for Model
  Explanation*, ICML. [arXiv:1908.08474](https://arxiv.org/abs/1908.08474)
- Sundararajan, Taly & Yan (2017), *Axiomatic Attribution for Deep Networks*
  (Integrated Gradients; the "completeness" property relative to a baseline),
  ICML. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)

**Identifiability in additive / neural-additive models.** GAM components are
only defined up to constants, and the standard fix is a centring (sum-to-zero)
constraint. A point constraint `f_b(x0) = 0` is the anchored alternative.
- Lengerich, Tan, Chang, Hooker & Caruana (2020), *Purifying Interaction
  Effects with the Functional ANOVA: An Efficient Algorithm for Recovering
  Identifiable Additive Models*, AISTATS.
  [arXiv:1911.04974](https://arxiv.org/abs/1911.04974) (mass can be moved
  between components without changing predictions; same issue, interaction
  level)
- Agarwal et al. (2021), *Neural Additive Models: Interpretable Machine
  Learning with Neural Nets*, NeurIPS.
  [arXiv:2004.13912](https://arxiv.org/abs/2004.13912) (one net per feature,
  summed; the closest ML analogue of the PANN)
- Wood (2017), *Generalized Additive Models: An Introduction with R*, 2nd ed.,
  CRC (identifiability constraints in `mgcv`).

**Gating = hurdle / trigger models.**
- Mullahy (1986), *Specification and testing of some modified count data
  models*, J. Econometrics 33, 341–365.
  [doi:10.1016/0304-4076(86)90002-3](https://doi.org/10.1016/0304-4076(86)90002-3)
  (hurdle models: a separate "is it zero?" stage)
- *Improvement of a neural network convection scheme by including triggering
  and evaluation in present and future climates* (2025), ARP-GEM.
  [arXiv:2511.05074](https://arxiv.org/abs/2511.05074) (an MLP trigger
  classifier sets tendencies to exactly 0 when deep convection is inactive)
- Sarauer et al. (2025), *A physics-informed machine learning parameterization
  for cloud microphysics in ICON*.
  [pdf](https://elib.dlr.de/216101/1/a-physics-informed-machine-learning-parameterization-for-cloud-microphysics-in-icon.pdf)
  (trigger classifier plus regressor, zero vector when inactive)
- Zhong et al. (2024), *Machine learning parameterization of the multi-scale
  Kain–Fritsch convection scheme…*, GMD 17, 3667.
  [link](https://gmd.copernicus.org/articles/17/3667/2024/gmd-17-3667-2024.html)

These ML-parameterization papers gate for physical realism and stability, not
for attribution. Using the gate or the anchor to make **per-process mean
attribution identifiable** in an additive model appears to be the new angle
here.
