import jax.numpy as jnp

from gaussian_fisher.energy import energy, horizontal_residual
from gaussian_fisher.geodesic import lifted_geodesic, projected_geodesic_path
from gaussian_fisher.solvers import minimize_gauge


def test_identity_endpoint_is_constant_for_dimensions_one_to_three():
    ts = jnp.linspace(0.0, 1.0, 5)

    for dimension in (1, 2, 3):
        mu = jnp.zeros(dimension)
        sigma = jnp.eye(dimension)
        result = minimize_gauge(mu, sigma, tolerance=1e-10)
        path = projected_geodesic_path(mu, sigma, ts, omega=result.omega_star)
        gauges = lifted_geodesic(mu, sigma, result.omega_star, ts)

        assert jnp.allclose(result.omega_star, jnp.zeros((dimension, dimension)), atol=1e-10)
        assert jnp.allclose(result.energy, 0.0, atol=1e-12)
        assert jnp.allclose(energy(mu, sigma, result.omega_vec), 0.0, atol=1e-12)
        assert jnp.allclose(horizontal_residual(mu, sigma, result.omega_vec), jnp.zeros_like(result.omega_vec), atol=1e-12)
        assert jnp.allclose(gauges, jnp.broadcast_to(jnp.eye(2 * dimension + 1), gauges.shape), atol=1e-10)
        assert jnp.allclose(path.mu, jnp.zeros((ts.shape[0], dimension)), atol=1e-10)
        assert jnp.allclose(path.sigma, jnp.broadcast_to(jnp.eye(dimension), path.sigma.shape), atol=1e-10)

