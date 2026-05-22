import jax.numpy as jnp
import jax.random as jr

from gaussian_fisher.energy import explicit_gradient, hessian_vector_autodiff, hessian_vector_explicit, hessian_vector_product
from gaussian_fisher.linalg import skew_from_vector


def test_hessian_vector_product_returns_skew_matrix():
    mu = jnp.array([0.2, -0.1])
    sigma = jnp.array([[1.3, 0.1], [0.1, 1.8]])
    omega = skew_from_vector(jnp.array([0.05]), 2)
    tangent = skew_from_vector(jnp.array([0.2]), 2)

    hvp = hessian_vector_product(mu, sigma, omega, tangent)

    assert hvp.shape == (2, 2)
    assert jnp.allclose(hvp + hvp.T, jnp.zeros((2, 2)), atol=1e-8)


def test_explicit_hessian_vector_matches_jax_jvp_and_finite_difference():
    mu = jnp.array([0.2, -0.1])
    sigma = jnp.array([[1.3, 0.1], [0.1, 1.8]])
    omega_vec = jnp.array([0.05])
    xi_vec = jnp.array([0.2])
    eps = 1e-5

    explicit = hessian_vector_explicit(mu, sigma, omega_vec, xi_vec)
    autodiff = hessian_vector_autodiff(mu, sigma, omega_vec, xi_vec)
    finite_difference = (
        explicit_gradient(mu, sigma, omega_vec + eps * xi_vec)
        - explicit_gradient(mu, sigma, omega_vec - eps * xi_vec)
    ) / (2.0 * eps)

    assert jnp.allclose(explicit, autodiff, atol=2e-7)
    assert jnp.allclose(explicit, finite_difference, atol=2e-7)


def test_explicit_hessian_vector_matches_jvp_for_random_dimensions():
    for dimension in (2, 3):
        key = jr.PRNGKey(200 + dimension)
        k_mu, k_sigma, k_omega, k_xi = jr.split(key, 4)
        size = dimension * (dimension - 1) // 2
        raw = jr.normal(k_sigma, (dimension, dimension)) * 0.2
        sigma = raw @ raw.T + jnp.eye(dimension)
        mu = jr.normal(k_mu, (dimension,)) * 0.2
        omega_vec = jr.normal(k_omega, (size,)) * 0.1
        xi_vec = jr.normal(k_xi, (size,)) * 0.1

        explicit = hessian_vector_explicit(mu, sigma, omega_vec, xi_vec)
        autodiff = hessian_vector_autodiff(mu, sigma, omega_vec, xi_vec)
        finite_differences = [
            (explicit_gradient(mu, sigma, omega_vec + eps * xi_vec) - explicit_gradient(mu, sigma, omega_vec - eps * xi_vec))
            / (2.0 * eps)
            for eps in (1e-4, 3e-5, 1e-5)
        ]

        assert jnp.allclose(explicit, autodiff, atol=5e-8)
        assert jnp.min(jnp.asarray([jnp.linalg.norm(fd - explicit) for fd in finite_differences])) < 5e-7
