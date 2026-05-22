import jax.numpy as jnp
import jax.random as jr

from gaussian_fisher.geodesic import projected_geodesic_path
from gaussian_fisher.linalg import matrix_power_spd
from gaussian_fisher.solvers import minimize_gauge


def test_zero_mean_projected_path_matches_spd_power_curve_for_diagonal_covariance():
    mu = jnp.zeros(2)
    sigma = jnp.diag(jnp.array([4.0, 9.0]))
    t = jnp.linspace(0.0, 1.0, 5)

    path = projected_geodesic_path(mu, sigma, t, omega=jnp.zeros((2, 2)))
    expected_sigmas = jnp.stack([matrix_power_spd(sigma, float(time)) for time in t])

    assert jnp.allclose(path.mu, jnp.zeros((5, 2)), atol=1e-6)
    assert jnp.allclose(path.sigma, expected_sigmas, atol=1e-5)


def test_zero_mean_random_spd_path_matches_spd_power_curve_tightly():
    key = jr.PRNGKey(17)
    raw = jr.normal(key, (3, 3))
    sigma = raw @ raw.T + 0.5 * jnp.eye(3)
    mu = jnp.zeros(3)
    ts = jnp.linspace(0.0, 1.0, 7)

    result = minimize_gauge(mu, sigma, tolerance=1e-10)
    path = projected_geodesic_path(mu, sigma, ts, omega=result.omega_star)
    expected_sigmas = jnp.stack([matrix_power_spd(sigma, float(t)) for t in ts])

    assert result.success_flag
    assert jnp.allclose(result.omega_star, jnp.zeros((3, 3)), atol=1e-10)
    assert jnp.allclose(path.mu, jnp.zeros((ts.shape[0], 3)), atol=1e-10)
    assert jnp.allclose(path.sigma, expected_sigmas, atol=1e-8)
