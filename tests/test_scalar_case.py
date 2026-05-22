import jax.numpy as jnp

from gaussian_fisher.geodesic import projected_geodesic_path
from gaussian_fisher.ode_checks import residual_summary
from gaussian_fisher.ps_reduction import gaussian_to_half_plane, half_plane_to_gaussian, scalar_fisher_distance


def test_scalar_half_plane_roundtrip():
    mu = jnp.array(1.25)
    sigma = jnp.array(2.5)

    x, y = gaussian_to_half_plane(mu, sigma)
    recovered_mu, recovered_sigma = half_plane_to_gaussian(x, y)

    assert jnp.allclose(recovered_mu, mu)
    assert jnp.allclose(recovered_sigma, sigma)


def test_scalar_zero_mean_distance_matches_log_covariance_formula():
    sigma0 = jnp.array(1.0)
    sigma1 = jnp.array(4.0)

    distance = scalar_fisher_distance(jnp.array(0.0), sigma0, jnp.array(0.0), sigma1)

    assert jnp.allclose(distance, jnp.sqrt(0.5) * jnp.abs(jnp.log(sigma1 / sigma0)), atol=1e-6)


def test_scalar_projected_path_lies_on_upper_half_plane_semicircle():
    mu = jnp.array([1.2])
    variance = jnp.array([[2.25]])
    ts = jnp.linspace(0.0, 1.0, 101)

    path = projected_geodesic_path(mu, variance, ts, omega=jnp.zeros((1, 1)))
    x_path = path.mu[:, 0] / jnp.sqrt(2.0)
    y_path = jnp.sqrt(path.sigma[:, 0, 0])
    x_end = mu[0] / jnp.sqrt(2.0)
    y_end = jnp.sqrt(variance[0, 0])
    center = (x_end * x_end + y_end * y_end - 1.0) / (2.0 * x_end)
    radius = jnp.sqrt(center * center + 1.0)
    summary = residual_summary(path.mu, path.sigma, ts)

    assert jnp.allclose(path.mu[0], jnp.zeros(1), atol=1e-10)
    assert jnp.allclose(path.sigma[0], jnp.ones((1, 1)), atol=1e-10)
    assert jnp.allclose(path.mu[-1], mu, atol=1e-9)
    assert jnp.allclose(path.sigma[-1], variance, atol=1e-9)
    assert jnp.allclose((x_path - center) ** 2 + y_path**2, radius**2, atol=2e-12)
    assert summary.max_mu_abs < 1e-4
    assert summary.max_sigma_abs < 1e-4


def test_scalar_zero_mean_path_is_vertical_in_upper_half_plane():
    variance = jnp.array([[4.0]])
    path = projected_geodesic_path(jnp.zeros(1), variance, jnp.linspace(0.0, 1.0, 21), omega=jnp.zeros((1, 1)))

    assert jnp.allclose(path.mu[:, 0] / jnp.sqrt(2.0), jnp.zeros(21), atol=1e-12)
