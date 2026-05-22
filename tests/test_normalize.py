import jax.numpy as jnp

from gaussian_fisher.normalize import denormalize_point, normalize_endpoint


def test_normalize_endpoint_and_denormalize_target_roundtrip():
    mu0 = jnp.array([1.0, -2.0])
    sigma0 = jnp.array([[2.0, 0.3], [0.3, 1.5]])
    mu1 = jnp.array([1.5, 0.25])
    sigma1 = jnp.array([[1.2, 0.1], [0.1, 3.0]])

    endpoint = normalize_endpoint(mu0, sigma0, mu1, sigma1)
    recovered_mu, recovered_sigma = denormalize_point(endpoint, endpoint.mu, endpoint.sigma)

    assert jnp.allclose(recovered_mu, mu1, atol=1e-6)
    assert jnp.allclose(recovered_sigma, sigma1, atol=1e-6)


def test_normalized_source_is_zero_identity():
    mu0 = jnp.array([0.2, 0.5])
    sigma0 = jnp.array([[1.7, 0.2], [0.2, 1.1]])
    endpoint = normalize_endpoint(mu0, sigma0, mu0, sigma0)

    assert jnp.allclose(endpoint.mu, jnp.zeros(2), atol=1e-6)
    assert jnp.allclose(endpoint.sigma, jnp.eye(2), atol=1e-6)

