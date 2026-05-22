import jax.numpy as jnp

from gaussian_fisher.ode import geodesic_acceleration, integrate_geodesic_ode
from gaussian_fisher.ode_checks import max_precision_residual_norm, max_residual_norm


def test_geodesic_acceleration_zero_velocity_is_zero():
    mu_dot = jnp.zeros(2)
    sigma = jnp.eye(2)
    sigma_dot = jnp.zeros((2, 2))

    mu_ddot, sigma_ddot = geodesic_acceleration(mu_dot, sigma, sigma_dot)

    assert jnp.allclose(mu_ddot, jnp.zeros(2))
    assert jnp.allclose(sigma_ddot, jnp.zeros((2, 2)))


def test_zero_mean_scalar_power_path_satisfies_ode_to_finite_difference_accuracy():
    t = jnp.linspace(0.0, 1.0, 101)
    a = jnp.log(3.0)
    sigmas = jnp.exp(a * t)[:, None, None]
    mus = jnp.zeros((t.shape[0], 1))

    assert max_residual_norm(mus, sigmas, t) < 5e-3


def test_diffrax_integrates_zero_mean_scalar_power_path():
    ts = jnp.linspace(0.0, 1.0, 11)
    a = jnp.log(3.0)
    initial_state = (
        jnp.zeros(1),
        jnp.ones((1, 1)),
        jnp.zeros(1),
        jnp.array([[a]]),
    )

    solution = integrate_geodesic_ode(initial_state, ts, dt0=1e-3)
    mus, sigmas, mu_dots, sigma_dots = solution.ys

    assert jnp.allclose(mus, jnp.zeros_like(mus), atol=1e-7)
    assert jnp.allclose(mu_dots, jnp.zeros_like(mu_dots), atol=1e-7)
    assert jnp.allclose(sigmas[:, 0, 0], jnp.exp(a * ts), atol=5e-6)
    assert jnp.allclose(sigma_dots[:, 0, 0], a * jnp.exp(a * ts), atol=5e-6)


def test_zero_mean_scalar_precision_path_satisfies_precision_ode():
    ts = jnp.linspace(0.0, 1.0, 101)
    a = jnp.log(3.0)
    deltas = jnp.zeros((ts.shape[0], 1))
    thetas = jnp.exp(-a * ts)[:, None, None]

    assert max_precision_residual_norm(deltas, thetas, ts) < 2e-3
