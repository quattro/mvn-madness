import jax.numpy as jnp

from gaussian_fisher.geodesic import (
    gaussian_geodesic_precision,
    project_gauge_matrix,
    project_to_precision_gaussian,
    projected_geodesic_path,
    projected_precision_geodesic_path,
)
from gaussian_fisher.kobayashi import gauge_matrix
from gaussian_fisher.linalg import inv_spd


def test_project_gauge_matrix_recovers_endpoint_coordinates():
    mu = jnp.array([0.4, -0.1])
    sigma = jnp.array([[1.4, 0.2], [0.2, 1.1]])
    omega = jnp.array([[0.0, 0.05], [-0.05, 0.0]])

    recovered_mu, recovered_sigma = project_gauge_matrix(gauge_matrix(mu, sigma, omega), 2)

    assert jnp.allclose(recovered_mu, mu, atol=1e-6)
    assert jnp.allclose(recovered_sigma, sigma, atol=1e-6)


def test_precision_projection_matches_target_theta_and_delta():
    mu = jnp.array([0.4, -0.1])
    sigma = jnp.array([[1.4, 0.2], [0.2, 1.1]])
    omega = jnp.array([[0.0, 0.05], [-0.05, 0.0]])

    delta, theta = project_to_precision_gaussian(gauge_matrix(mu, sigma, omega), 2)
    target_theta = inv_spd(sigma)

    assert jnp.allclose(theta, target_theta, atol=1e-8)
    assert jnp.allclose(delta, target_theta @ mu, atol=1e-8)


def test_projected_path_hits_normalized_endpoints():
    mu = jnp.array([0.25, 0.1])
    sigma = jnp.array([[1.5, 0.1], [0.1, 1.2]])
    t = jnp.array([0.0, 1.0])

    path = projected_geodesic_path(mu, sigma, t, omega=jnp.zeros((2, 2)))

    assert jnp.allclose(path.mu[0], jnp.zeros(2), atol=1e-6)
    assert jnp.allclose(path.sigma[0], jnp.eye(2), atol=1e-6)
    assert jnp.allclose(path.mu[-1], mu, atol=1e-5)
    assert jnp.allclose(path.sigma[-1], sigma, atol=1e-5)


def test_projected_precision_path_hits_normalized_precision_endpoints():
    mu = jnp.array([0.25, 0.1])
    sigma = jnp.array([[1.5, 0.1], [0.1, 1.2]])
    t = jnp.array([0.0, 1.0])

    path = projected_precision_geodesic_path(mu, sigma, t, omega=jnp.zeros((2, 2)))
    target_theta = inv_spd(sigma)

    assert jnp.allclose(path.delta[0], jnp.zeros(2), atol=1e-10)
    assert jnp.allclose(path.theta[0], jnp.eye(2), atol=1e-10)
    assert jnp.allclose(path.delta[-1], target_theta @ mu, atol=1e-8)
    assert jnp.allclose(path.theta[-1], target_theta, atol=1e-8)


def test_gaussian_geodesic_precision_returns_normalized_precision_coordinates():
    mu0 = jnp.array([1.0, -0.5])
    sigma0 = jnp.array([[2.0, 0.1], [0.1, 1.3]])
    mu1 = jnp.array([1.2, 0.25])
    sigma1 = jnp.array([[1.4, 0.2], [0.2, 1.6]])
    path = gaussian_geodesic_precision(mu0, sigma0, mu1, sigma1, jnp.array([0.0]))

    assert jnp.allclose(path.delta[0], jnp.zeros(2), atol=1e-10)
    assert jnp.allclose(path.theta[0], jnp.eye(2), atol=1e-10)
