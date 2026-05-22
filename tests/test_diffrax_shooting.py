import jax.numpy as jnp
import jax.random as jr

from gaussian_fisher.geodesic import projected_geodesic_path
from gaussian_fisher.ode import integrate_geodesic_ode
from gaussian_fisher.solvers import minimize_gauge


def test_diffrax_shooting_from_horizontal_lift_velocity_lands_near_target():
    key = jr.PRNGKey(611)
    raw = jr.normal(key, (2, 2)) * 0.1
    sigma = raw @ raw.T + jnp.eye(2)
    mu = jr.normal(key, (2,)) * 0.1
    result = minimize_gauge(mu, sigma, tolerance=1e-10, max_steps=512)
    h = 1e-5
    short_path = projected_geodesic_path(mu, sigma, jnp.array([0.0, h]), omega=result.omega_star)
    initial_mu_dot = (short_path.mu[1] - short_path.mu[0]) / h
    initial_sigma_dot = (short_path.sigma[1] - short_path.sigma[0]) / h

    solution = integrate_geodesic_ode(
        (jnp.zeros(2), jnp.eye(2), initial_mu_dot, initial_sigma_dot),
        jnp.linspace(0.0, 1.0, 21),
        dt0=1e-3,
    )
    mus, sigmas, _, _ = solution.ys

    assert jnp.linalg.norm(mus[-1] - mu) < 5e-7
    assert jnp.linalg.norm(sigmas[-1] - sigma) < 5e-7

