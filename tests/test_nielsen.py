import jax.numpy as jnp
import jax.random as jr

from gaussian_fisher.linalg import invsqrtm_spd, is_spd, logm_spd
from gaussian_fisher.nielsen import (
    expectation_curve,
    expectation_to_source,
    jeffreys_mvn,
    kl_mvn,
    natural_curve,
    natural_to_source,
    nielsen_distance,
    source_curve,
    source_to_expectation,
    source_to_natural,
)
from gaussian_fisher.ps_reduction import scalar_fisher_distance


def _random_spd(seed: int, dimension: int) -> jnp.ndarray:
    raw = jr.normal(jr.PRNGKey(seed), (dimension, dimension))
    return raw @ raw.T + 0.5 * jnp.eye(dimension)


def test_kl_and_jeffreys_identities():
    mu0 = jnp.array([0.1, -0.2])
    sigma0 = jnp.array([[1.3, 0.1], [0.1, 0.9]])
    mu1 = jnp.array([0.4, 0.2])
    sigma1 = jnp.array([[1.8, 0.2], [0.2, 1.1]])

    assert jnp.allclose(kl_mvn(mu0, sigma0, mu0, sigma0), 0.0, atol=1e-12)
    assert kl_mvn(mu0, sigma0, mu1, sigma1) >= 0.0
    assert jnp.allclose(jeffreys_mvn(mu0, sigma0, mu0, sigma0), 0.0, atol=1e-12)
    assert jnp.allclose(
        jeffreys_mvn(mu0, sigma0, mu1, sigma1),
        jeffreys_mvn(mu1, sigma1, mu0, sigma0),
        atol=1e-12,
    )


def test_natural_and_expectation_round_trips():
    mu = jnp.array([0.3, -0.5, 0.2])
    sigma = _random_spd(10, 3)

    h, k = source_to_natural(mu, sigma)
    recovered_mu, recovered_sigma = natural_to_source(h, k)
    assert jnp.allclose(recovered_mu, mu, atol=1e-10)
    assert jnp.allclose(recovered_sigma, sigma, atol=1e-10)

    m, big_m = source_to_expectation(mu, sigma)
    recovered_mu, recovered_sigma = expectation_to_source(m, big_m)
    assert jnp.allclose(recovered_mu, mu, atol=1e-12)
    assert jnp.allclose(recovered_sigma, sigma, atol=1e-12)


def test_curve_endpoints_and_spd_covariances():
    mu0 = jnp.array([0.1, -0.2])
    sigma0 = _random_spd(20, 2)
    mu1 = jnp.array([0.7, 0.4])
    sigma1 = _random_spd(21, 2)
    ts = jnp.linspace(0.0, 1.0, 9)

    for curve in (source_curve, natural_curve, expectation_curve):
        mus, sigmas = curve(mu0, sigma0, mu1, sigma1, ts)
        assert jnp.allclose(mus[0], mu0, atol=1e-10)
        assert jnp.allclose(sigmas[0], sigma0, atol=1e-10)
        assert jnp.allclose(mus[-1], mu1, atol=1e-10)
        assert jnp.allclose(sigmas[-1], sigma1, atol=1e-10)
        assert all(is_spd(sigmas[index], tol=1e-10) for index in range(sigmas.shape[0]))


def test_scalar_nielsen_approximation_converges_to_exact_distance():
    mu0 = jnp.array([0.0])
    sigma0 = jnp.array([[1.0]])
    mu1 = jnp.array([0.8])
    sigma1 = jnp.array([[2.25]])
    exact = scalar_fisher_distance(mu0[0], sigma0[0, 0], mu1[0], sigma1[0, 0])

    coarse = nielsen_distance(mu0, sigma0, mu1, sigma1, 8, curve="source")
    fine = nielsen_distance(mu0, sigma0, mu1, sigma1, 512, curve="source")

    assert abs(fine.distance - exact) < abs(coarse.distance - exact)
    assert jnp.allclose(fine.distance, exact, atol=3e-3)


def test_zero_mean_covariance_nielsen_converges_to_exact_distance():
    mu0 = jnp.zeros(2)
    mu1 = jnp.zeros(2)
    sigma0 = jnp.eye(2)
    sigma1 = jnp.array([[3.0, 0.4], [0.4, 1.5]])
    normalized = invsqrtm_spd(sigma0) @ sigma1 @ invsqrtm_spd(sigma0)
    exact = jnp.linalg.norm(logm_spd(normalized)) / jnp.sqrt(2.0)

    coarse = nielsen_distance(mu0, sigma0, mu1, sigma1, 8, curve="natural")
    fine = nielsen_distance(mu0, sigma0, mu1, sigma1, 512, curve="natural")

    assert abs(fine.distance - exact) < abs(coarse.distance - exact)
    assert jnp.allclose(fine.distance, exact, atol=2e-3)
