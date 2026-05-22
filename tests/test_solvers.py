import jax.numpy as jnp

from gaussian_fisher.solvers import solve_horizontal_gauge
from gaussian_fisher.solvers import minimize_gauge, solve_horizontal_root


def test_solver_converges_immediately_for_zero_mean_diagonal_covariance():
    mu = jnp.zeros(2)
    sigma = jnp.diag(jnp.array([2.0, 3.0]))

    result = solve_horizontal_gauge(mu, sigma)

    assert result.converged
    assert jnp.allclose(result.omega, jnp.zeros((2, 2)), atol=1e-8)
    assert result.gradient_norm <= 1e-8


def test_minimize_gauge_returns_structured_horizontal_result():
    mu = jnp.zeros(2)
    sigma = jnp.diag(jnp.array([2.0, 3.0]))

    result = minimize_gauge(mu, sigma)

    assert result.success_flag
    assert result.horizontal_norm <= 1e-8
    assert result.gradient_norm <= 1e-8


def test_horizontal_root_matches_minimization_for_zero_mean_case():
    mu = jnp.zeros(2)
    sigma = jnp.diag(jnp.array([2.0, 3.0]))

    minimized = minimize_gauge(mu, sigma)
    rooted = solve_horizontal_root(mu, sigma)

    assert rooted.success_flag
    assert jnp.allclose(rooted.omega_star, minimized.omega_star, atol=1e-8)
