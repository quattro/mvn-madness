import jax.numpy as jnp
import jax.random as jr

from gaussian_fisher.expmap import covariance_only_expmap, expmap, expmap_path
from gaussian_fisher.geodesic import projected_geodesic_path
from gaussian_fisher.linalg import expm_sym
from gaussian_fisher.ode_checks import max_residual_norm
from gaussian_fisher.solvers import minimize_gauge


def test_expmap_zero_tangent_returns_base_point():
    mu = jnp.array([0.4, -0.2])
    sigma = jnp.array([[1.7, 0.2], [0.2, 0.9]])
    v_mu = jnp.zeros(2)
    v_sigma = jnp.zeros((2, 2))

    mu_t, sigma_t = expmap(mu, sigma, v_mu, v_sigma, t=1.0)

    assert jnp.allclose(mu_t, mu, atol=1e-12)
    assert jnp.allclose(sigma_t, sigma, atol=1e-12)


def test_expmap_initial_derivative_matches_tangent():
    mu = jnp.array([0.2, -0.5])
    sigma = jnp.array([[2.0, 0.3], [0.3, 1.4]])
    v_mu = jnp.array([0.1, -0.07])
    v_sigma = jnp.array([[0.05, 0.02], [0.02, -0.03]])
    h = 1e-6

    mu_h, sigma_h = expmap(mu, sigma, v_mu, v_sigma, t=h)

    assert jnp.allclose((mu_h - mu) / h, v_mu, atol=2e-6, rtol=2e-6)
    assert jnp.allclose((sigma_h - sigma) / h, v_sigma, atol=2e-6, rtol=2e-6)


def test_scalar_expmap_path_satisfies_geodesic_ode_and_stays_spd():
    mu = jnp.array([0.3])
    sigma = jnp.array([[1.4]])
    v_mu = jnp.array([0.15])
    v_sigma = jnp.array([[0.05]])
    ts = jnp.linspace(0.0, 1.0, 201)

    mus, sigmas = expmap_path(mu, sigma, v_mu, v_sigma, ts)

    assert jnp.all(sigmas[:, 0, 0] > 0.0)
    assert max_residual_norm(mus, sigmas, ts) < 2e-6


def test_covariance_only_expmap_matches_spd_affine_invariant_geodesic():
    mu = jnp.array([0.1, -0.2])
    sigma = jnp.array([[2.0, 0.4], [0.4, 1.3]])
    v_sigma = jnp.array([[0.08, -0.03], [-0.03, 0.04]])
    sigma_sqrt = jnp.linalg.eigh(sigma)[1] @ jnp.diag(jnp.sqrt(jnp.linalg.eigh(sigma)[0])) @ jnp.linalg.eigh(sigma)[1].T
    sigma_invsqrt = jnp.linalg.eigh(sigma)[1] @ jnp.diag(1.0 / jnp.sqrt(jnp.linalg.eigh(sigma)[0])) @ jnp.linalg.eigh(sigma)[1].T
    normalized_tangent = sigma_invsqrt @ v_sigma @ sigma_invsqrt

    mu_t, sigma_t = expmap(mu, sigma, jnp.zeros(2), v_sigma, t=0.75)
    ref_mu, ref_sigma = covariance_only_expmap(mu, sigma, v_sigma, t=0.75)
    expected_sigma = sigma_sqrt @ expm_sym(0.75 * normalized_tangent) @ sigma_sqrt

    assert jnp.allclose(mu_t, mu, atol=1e-12)
    assert jnp.allclose(ref_mu, mu, atol=1e-12)
    assert jnp.allclose(sigma_t, expected_sigma, atol=1e-10)
    assert jnp.allclose(ref_sigma, expected_sigma, atol=1e-10)


def test_expmap_path_satisfies_gaussian_geodesic_ode():
    mu = jnp.array([0.2, -0.1])
    sigma = jnp.array([[2.0, 0.3], [0.3, 1.5]])
    v_mu = jnp.array([0.1, 0.2])
    v_sigma = jnp.array([[0.05, 0.02], [0.02, -0.03]])
    ts = jnp.linspace(0.0, 1.0, 201)

    mus, sigmas = expmap_path(mu, sigma, v_mu, v_sigma, ts)

    assert max_residual_norm(mus, sigmas, ts) < 5e-7


def test_expmap_agrees_with_horizontal_lift_endpoint_from_estimated_initial_velocity():
    mu_target = jnp.array([0.25, -0.15])
    sigma_target = jnp.array([[1.35, 0.12], [0.12, 0.85]])
    result = minimize_gauge(mu_target, sigma_target, tolerance=1e-10)
    ts = jnp.linspace(0.0, 1.0, 401)
    path = projected_geodesic_path(mu_target, sigma_target, ts, omega=result.omega_star)
    h = ts[1] - ts[0]
    v_mu = (-3.0 * path.mu[0] + 4.0 * path.mu[1] - path.mu[2]) / (2.0 * h)
    v_sigma = (-3.0 * path.sigma[0] + 4.0 * path.sigma[1] - path.sigma[2]) / (2.0 * h)

    mu_1, sigma_1 = expmap(jnp.zeros(2), jnp.eye(2), v_mu, v_sigma, t=1.0)

    assert result.success_flag
    assert jnp.allclose(mu_1, mu_target, atol=2e-5)
    assert jnp.allclose(sigma_1, sigma_target, atol=2e-5)


def test_expmap_path_vectorizes_over_random_times():
    key = jr.PRNGKey(0)
    raw = jr.normal(key, (2, 2))
    sigma = raw @ raw.T + 0.5 * jnp.eye(2)
    ts = jnp.array([0.0, 0.2, 0.7])

    mus, sigmas = expmap_path(jnp.zeros(2), sigma, jnp.array([0.05, -0.01]), 0.02 * jnp.eye(2), ts)

    assert mus.shape == (3, 2)
    assert sigmas.shape == (3, 2, 2)
    assert jnp.all(jnp.linalg.eigvalsh(sigmas) > 0.0)
