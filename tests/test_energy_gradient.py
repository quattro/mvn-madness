import jax.numpy as jnp
import jax.random as jr

from gaussian_fisher.energy import autodiff_gradient, energy, explicit_energy_gradient_matrix, explicit_gradient
from gaussian_fisher.linalg import skew_from_vector, vector_from_skew


def test_explicit_energy_gradient_matches_jax_grad_in_skew_coordinates():
    mu = jnp.array([0.35, -0.2])
    sigma = jnp.array([[1.8, 0.25], [0.25, 1.2]])
    omega_vec = jnp.array([0.15])

    autodiff = autodiff_gradient(mu, sigma, omega_vec)
    explicit = explicit_gradient(mu, sigma, omega_vec)

    assert jnp.allclose(autodiff, explicit, atol=2e-8)


def test_explicit_energy_gradient_matches_finite_difference_pairing():
    mu = jnp.array([0.35, -0.2])
    sigma = jnp.array([[1.8, 0.25], [0.25, 1.2]])
    omega = skew_from_vector(jnp.array([0.15]), 2)
    xi = skew_from_vector(jnp.array([-0.4]), 2)
    eps = 1e-5

    finite_difference = (energy(mu, sigma, omega + eps * xi) - energy(mu, sigma, omega - eps * xi)) / (2.0 * eps)
    pairing = -jnp.trace(explicit_energy_gradient_matrix(mu, sigma, omega) @ xi)

    assert jnp.allclose(finite_difference, pairing, atol=2e-7)


def test_explicit_energy_gradient_matches_finite_differences_for_random_dimensions():
    for dimension in (2, 3):
        key = jr.PRNGKey(100 + dimension)
        k_mu, k_sigma, k_omega, k_xi = jr.split(key, 4)
        raw = jr.normal(k_sigma, (dimension, dimension)) * 0.3
        sigma = raw @ raw.T + jnp.eye(dimension)
        mu = jr.normal(k_mu, (dimension,)) * 0.3
        omega = skew_from_vector(jr.normal(k_omega, (dimension * (dimension - 1) // 2,)) * 0.1, dimension)
        xi = skew_from_vector(jr.normal(k_xi, (dimension * (dimension - 1) // 2,)) * 0.1, dimension)
        omega_vec = vector_from_skew(omega)

        finite_differences = [
            (energy(mu, sigma, omega + eps * xi) - energy(mu, sigma, omega - eps * xi)) / (2.0 * eps)
            for eps in (1e-4, 3e-5, 1e-5)
        ]
        pairing = -jnp.trace(explicit_energy_gradient_matrix(mu, sigma, omega) @ xi)

        assert jnp.min(jnp.abs(jnp.asarray(finite_differences) - pairing)) < 5e-7
        assert jnp.allclose(explicit_gradient(mu, sigma, omega_vec), autodiff_gradient(mu, sigma, omega_vec), atol=5e-8)


def test_explicit_gradient_is_zero_for_zero_mean_diagonal_covariance():
    mu = jnp.zeros(2)
    sigma = jnp.diag(jnp.array([2.0, 3.0]))
    omega = jnp.zeros((2, 2))

    explicit = explicit_gradient(mu, sigma, jnp.array([0.0]))

    assert jnp.allclose(explicit, jnp.zeros_like(explicit), atol=1e-7)
