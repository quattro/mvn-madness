import jax.numpy as jnp
import jax.random as jr

from gaussian_fisher.energy import hessian_vector_autodiff
from gaussian_fisher.geodesic import projected_geodesic_path
from gaussian_fisher.linalg import is_spd
from gaussian_fisher.ode_checks import residual_summary
from gaussian_fisher.solvers import minimize_gauge


def _random_endpoint(seed: int, dimension: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jr.PRNGKey(seed)
    k_mu, k_sigma = jr.split(key)
    raw = jr.normal(k_sigma, (dimension, dimension)) * 0.2
    sigma = raw @ raw.T + jnp.eye(dimension)
    mu = jr.normal(k_mu, (dimension,)) * 0.2
    return mu, sigma


def test_horizontal_minimizer_recovers_endpoints_and_spd_path_for_random_dimensions():
    for dimension in (2, 3):
        mu, sigma = _random_endpoint(300 + dimension, dimension)
        ts = jnp.linspace(0.0, 1.0, 41)

        result = minimize_gauge(mu, sigma, tolerance=1e-9, max_steps=512)
        path = projected_geodesic_path(mu, sigma, ts, omega=result.omega_star)

        assert result.success_flag
        assert result.horizontal_norm < 1e-7
        assert jnp.allclose(path.mu[0], jnp.zeros(dimension), atol=1e-10)
        assert jnp.allclose(path.sigma[0], jnp.eye(dimension), atol=1e-10)
        assert jnp.allclose(path.mu[-1], mu, atol=1e-8)
        assert jnp.allclose(path.sigma[-1], sigma, atol=1e-8)
        assert all(is_spd(path.sigma[i], tol=1e-10) for i in range(ts.shape[0]))


def test_projected_horizontal_path_has_small_ode_residual_for_random_endpoint():
    mu, sigma = _random_endpoint(407, 2)
    ts = jnp.linspace(0.0, 1.0, 101)

    result = minimize_gauge(mu, sigma, tolerance=1e-10, max_steps=512)
    path = projected_geodesic_path(mu, sigma, ts, omega=result.omega_star)
    summary = residual_summary(path.mu, path.sigma, ts)

    assert result.horizontal_norm < 1e-8
    assert summary.max_mu_abs < 1e-5
    assert summary.max_sigma_abs < 1e-5
    assert summary.max_mu_rel < 2e-3
    assert summary.max_sigma_rel < 2e-3


def test_hessian_at_random_horizontal_minimizer_has_positive_minimum_eigenvalue():
    mu, sigma = _random_endpoint(509, 3)
    result = minimize_gauge(mu, sigma, tolerance=1e-9, max_steps=512)
    size = result.omega_vec.shape[0]
    basis = jnp.eye(size)
    hessian = jnp.stack([hessian_vector_autodiff(mu, sigma, result.omega_vec, basis[i]) for i in range(size)], axis=1)

    assert result.horizontal_norm < 1e-7
    assert jnp.min(jnp.linalg.eigvalsh(0.5 * (hessian + hessian.T))) > 0.0

