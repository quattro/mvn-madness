# Gaussian Fisher Geodesics

Numerical experiments for Fisher-Rao geodesics between multivariate normal distributions.

The package focuses on a reliable harness for testing candidate geodesic formulations. The initial implementation includes:

- affine endpoint normalization and denormalization,
- SPD linear algebra helpers implemented with JAX,
- Kobayashi-style gauge-lift matrices and energy,
- closed-form Fisher-Rao exponential maps for MVN initial-value problems,
- Riemannian gradient descent utilities using the closed-form exponential map,
- explicit horizontal gradient checks,
- simple projected gradient solvers,
- projected geodesic paths from matrix logarithms,
- finite-difference checks against the Gaussian geodesic ODE.

The primary numerical stack is JAX, Equinox, Optimistix, and Diffrax. Diffrax is used by optional shooting/validation experiments.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Closed-Form Exponential Map

The endpoint/log-map problem is handled by the horizontal-lift solver. The initial-value problem has a separate closed-form implementation:

```python
import jax.numpy as jnp

from gaussian_fisher.expmap import expmap, expmap_path

mu = jnp.array([0.0, 0.0])
sigma = jnp.eye(2)
v_mu = jnp.array([0.1, -0.05])
v_sigma = jnp.array([[0.02, 0.01], [0.01, -0.03]])

mu_1, sigma_1 = expmap(mu, sigma, v_mu, v_sigma, t=1.0)

ts = jnp.linspace(0.0, 1.0, 101)
mu_path, sigma_path = expmap_path(mu, sigma, v_mu, v_sigma, ts)
```

`expmap(...)` computes

```text
(mu, Sigma), (v_mu, v_Sigma), t -> (mu(t), Sigma(t))
```

It is an exponential map for a tangent vector, not an endpoint solver. Use `gaussian_fisher.geodesic` when the inputs are two endpoint distributions and the initial velocity is unknown.

The implementation:

- normalizes to `(0, I)` by affine invariance,
- evaluates the closed-form precision-coordinate solution with symmetric eigendecompositions,
- uses stable branches for `sinh(tg) / g` and `(cosh(tg) - 1) / g^2`,
- solves in precision coordinates instead of explicitly inverting for the mean.

Run the expmap tests directly:

```bash
pytest tests/test_expmap.py -q
```

## Riemannian Gradient Descent

`gaussian_fisher.riemannian_grad` converts Euclidean gradients to Fisher-Rao Riemannian gradients and takes descent steps with the closed-form exponential map.

```python
import jax.numpy as jnp

from gaussian_fisher.riemannian_grad import rgd

target = jnp.array([1.0, -0.5])

def objective(mu, sigma):
    del sigma
    diff = mu - target
    return 0.5 * diff @ diff

result = rgd(
    objective,
    mu0=jnp.zeros(2),
    sigma0=jnp.eye(2),
    step_size=0.05,
    n_steps=20,
    backtracking=True,
)
```

The conversion is:

```text
v_mu = Sigma @ grad_mu
v_Sigma = 2 Sigma @ sym(grad_Sigma) @ Sigma
```

For minimization, `rgd(...)` moves along the negative Riemannian gradient. Backtracking is enabled by default because the exponential map is exact for the tangent vector, but a large objective step can still increase the loss.

Run the RGD demo:

```bash
python experiments/riemannian_gradient_descent_demo.py \
  --step-size 0.05 \
  --n-steps 20
```

The demo prints initial/final objective values, accepted steps, final gradient norm, minimum eigenvalue of `Sigma`, and whether the objective decreased monotonically.

Run the focused RGD tests:

```bash
pytest tests/test_riemannian_grad.py -q
```

## Adversarial Scans

The default pytest suite stays fast. Larger searches for nonuniqueness, optimizer branch failures, and Hessian degeneracy live under `experiments/`.

### `d=2` Parameter Grid

Scans diagonal covariance ratios, mean norms, and mean angles. Each endpoint starts with an omega range of `[-50, 50]` and expands the range if the grid minimum or critical brackets are near the boundary.

```bash
python experiments/adversarial_d2_grid_scan.py \
  --points 1001 \
  --output experiments/output/adversarial_d2_grid.csv
```

Use `--limit` for a small smoke run:

```bash
python experiments/adversarial_d2_grid_scan.py \
  --points 101 \
  --limit 8 \
  --output experiments/output/adversarial_d2_grid_smoke.csv
```

### Random Multistart Scans

Generates random `d=3,4,5` endpoints with condition numbers up to `1e6` and mean norms up to `100`, then runs multiple random gauge initializations per endpoint.

```bash
python experiments/adversarial_random_multistart.py \
  --dimensions 3 4 5 \
  --endpoints 10 \
  --starts 50 \
  --output experiments/output/adversarial_random_multistart.csv
```

### Hessian Degeneracy Search

Runs a heuristic random search and keeps endpoints with the smallest Hessian minimum eigenvalues at the optimized gauge.

```bash
python experiments/hessian_degeneracy_search.py \
  --dimensions 3 4 5 \
  --samples 100 \
  --keep 25 \
  --output experiments/output/hessian_degeneracy_search.csv
```

### Summary Report

Summarize one or more adversarial CSV outputs:

```bash
python experiments/summarize_adversarial_scans.py \
  experiments/output/adversarial_d2_grid.csv \
  experiments/output/adversarial_random_multistart.csv \
  experiments/output/hessian_degeneracy_search.csv
```

The summary reports total endpoints, multiple apparent minima, multiple critical points, worst horizontal residual, worst endpoint error, smallest Hessian eigenvalue, largest optimizer spread, and the top 10 most suspicious rows.

Solver failures are recorded as data. Treat them as endpoints to inspect, not as reasons to discard a scan.

For ill-conditioned endpoints, prefer precision-coordinate diagnostics. The stable projected coordinates are `(delta, Theta)`, where `Theta = Sigma^{-1}` and `delta = Theta @ mu`. The covariance path API still returns `(mu, Sigma)`, but recovering `Sigma = Theta^{-1}` is ill-conditioned near the SPD boundary.

Available APIs:

- `gaussian_geodesic_precision(...)` returns the normalized precision-coordinate path `(delta(t), Theta(t))`.
- `gaussian_geodesic(...)` returns the covariance-coordinate path `(mu(t), Sigma(t))`.

Adversarial reports classify endpoint diagnostics as:

- `success`: lifted, precision, and covariance-coordinate endpoint errors are small;
- `projection_conditioning`: lifted and precision-coordinate errors are small, but covariance-coordinate error is amplified by inversion;
- `geometric_or_solver_failure`: lifted or precision-coordinate errors are large.

ODE diagnostics used in Nielsen/horizontal-lift comparisons are reported in precision coordinates. The covariance-coordinate residual functions remain available for special-case validation, but high-condition benchmark rows use the transformed `(delta, Theta)` residuals.

## Nielsen Approximation Baselines

This repository includes Nielsen-style Fisher-Rao MVN approximation baselines. These approximate the Fisher-Rao distance by discretizing a chosen curve between two MVNs and summing local segment lengths approximated by `sqrt(Jeffreys divergence)`. These baselines are not exact geodesic solvers, except in limiting or special cases.

Available curves:

- source parameters,
- natural parameters,
- expectation parameters.

The Calvo-Oller/SPD embedding curve is left unimplemented until the embedding convention is made explicit. Hilbert/SPD-cone distances are implemented separately as proxy distances, not Fisher-Rao distances.

Run a compact comparison:

```bash
python experiments/compare_nielsen.py
```

Generate a convergence table:

```bash
python experiments/nielsen_convergence.py \
  --dimension 3 \
  --n-steps 4 8 16 32 64 128 256 \
  --output experiments/output/nielsen_convergence.csv
```

Compare Nielsen baselines against the horizontal-lift solver on random endpoints:

```bash
python experiments/nielsen_vs_horizontal_lift.py \
  --named-cases scalar zero_mean d2_noncommuting_moderate d3_random_moderate projection_conditioning \
  --dimensions 2 3 4 \
  --endpoints 10 \
  --n-steps 16 32 64 128 \
  --output experiments/output/nielsen_vs_horizontal_lift.csv
```

## References

The code in this repository is an implementation and testing harness around the following sources.

- Shimpei Kobayashi, "Geodesics of multivariate normal distributions and a Toda lattice type Lax pair," Physica Scripta 98(11), 2023. DOI: [10.1088/1402-4896/ad0087](https://doi.org/10.1088/1402-4896/ad0087), arXiv: [2304.12575](https://arxiv.org/abs/2304.12575). This is the source for the block-Cholesky/Riemannian-submersion view behind the horizontal-lift endpoint solver.
- Miquel Calvo and Josep M. Oller, "A distance between multivariate normal distributions based in an embedding into the Siegel group," Journal of Multivariate Analysis 35(2), 223-242, 1990. DOI: [10.1016/0047-259X(90)90026-E](https://doi.org/10.1016/0047-259X(90)90026-E). This is the source for the Calvo-Oller SPD/Siegel embedding and lower-bound perspective used in the Nielsen comparison literature.
- Miquel Calvo and Josep M. Oller, "An Explicit Solution Of Information Geodesic Equations For The Multivariate Normal Model," Statistics & Risk Modeling 9(1-2), 119-138, 1991. DOI: [10.1524/strm.1991.9.12.119](https://doi.org/10.1524/strm.1991.9.12.119). This is one of the primary sources for the closed-form MVN geodesic initial-value solution implemented in `gaussian_fisher.expmap`.
- P. Svante Eriksen, "Geodesics connected with the Fisher metric on the multivariate normal manifold," 1987. This is the other primary source commonly cited for the explicit MVN Fisher geodesic initial-value problem.
- Frank Nielsen, "A Simple Approximation Method for the Fisher-Rao Distance between Multivariate Normal Distributions," Entropy 25(4):654, 2023. DOI: [10.3390/e25040654](https://doi.org/10.3390/e25040654), arXiv: [2302.08175](https://arxiv.org/abs/2302.08175). This is the source for the Nielsen-style discretized approximations using source, natural, and expectation parameter curves with `sqrt(Jeffreys divergence)` segment lengths.
- Julianna Pinele, Joao E. Strapasson, and Sueli I. R. Costa, "The Fisher-Rao Distance between Multivariate Normal Distributions: Special Cases, Bounds and Applications," Entropy 22(4):404, 2020. DOI: [10.3390/e22040404](https://doi.org/10.3390/e22040404). This survey is useful for the affine invariance, geodesic ODE, special-case closed forms, bounds, and the distinction between initial-value formulas and the harder endpoint distance problem.

## AI Assistance Disclosure

This repository was developed with significant assistance from OpenAI Codex and ChatGPT. The mathematical direction, implementation choices, tests, diagnostics, and documentation were iterated interactively with AI assistance; results should be reviewed and validated as research software rather than treated as an independently verified reference implementation.
