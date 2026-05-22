import jax.numpy as jnp

from gaussian_fisher.riemannian_grad import (
    euclidean_to_riemannian_gradient,
    rgd,
    riemannian_gradient_direction,
    riemannian_gradient_step,
)


def test_mean_quadratic_riemannian_gradient_direction_reduces_objective():
    target = jnp.array([1.0, -0.5])
    mu = jnp.array([0.2, 0.1])
    sigma = jnp.array([[1.5, 0.2], [0.2, 0.8]])

    def objective(mu_arg, sigma_arg):
        del sigma_arg
        diff = mu_arg - target
        return 0.5 * diff @ diff

    grad_mu = mu - target
    grad_sigma = jnp.zeros((2, 2))
    rgrad_mu, rgrad_sigma = euclidean_to_riemannian_gradient(mu, sigma, grad_mu, grad_sigma)
    direction_mu, direction_sigma = riemannian_gradient_direction(mu, sigma, grad_mu, grad_sigma)
    next_mu, next_sigma, value, grad_norm = riemannian_gradient_step(objective, mu, sigma, step_size=0.05)

    assert jnp.allclose(rgrad_mu, sigma @ grad_mu)
    assert jnp.allclose(rgrad_sigma, jnp.zeros((2, 2)))
    assert jnp.allclose(direction_mu, -sigma @ grad_mu)
    assert jnp.allclose(direction_sigma, jnp.zeros((2, 2)))
    assert objective(next_mu, next_sigma) < value
    assert grad_norm > 0.0


def test_covariance_objective_decreases_under_rgd():
    target = jnp.array([[1.2, 0.1], [0.1, 0.7]])
    target_inv = jnp.linalg.inv(target)
    mu0 = jnp.zeros(2)
    sigma0 = jnp.array([[2.0, 0.3], [0.3, 1.4]])

    def objective(mu_arg, sigma_arg):
        del mu_arg
        sign, logdet = jnp.linalg.slogdet(sigma_arg)
        return 0.5 * (jnp.trace(target_inv @ sigma_arg) - logdet) + 0.0 * sign

    result = rgd(objective, mu0, sigma0, step_size=0.05, n_steps=8)

    assert result.success
    assert result.values[-1] < result.values[0]
    assert jnp.all(jnp.diff(result.values) <= 1e-12)
    assert jnp.min(jnp.linalg.eigvalsh(result.sigma)) > 0.0


def test_mvn_negative_log_likelihood_decreases_with_backtracking():
    samples = jnp.array(
        [
            [0.9, -0.2],
            [1.1, -0.4],
            [0.7, 0.0],
            [1.3, -0.3],
            [0.8, -0.5],
        ]
    )
    mu0 = jnp.array([0.0, 0.0])
    sigma0 = jnp.array([[1.8, 0.2], [0.2, 1.1]])

    def objective(mu_arg, sigma_arg):
        centered = samples - mu_arg
        solved = jnp.linalg.solve(sigma_arg, centered.T).T
        sign, logdet = jnp.linalg.slogdet(sigma_arg)
        quadratic = jnp.sum(centered * solved)
        return 0.5 * samples.shape[0] * logdet + 0.5 * quadratic + 0.0 * sign

    result = rgd(objective, mu0, sigma0, step_size=0.1, n_steps=12, backtracking=True)

    assert result.success
    assert result.values[-1] < result.values[0]
    assert jnp.all(jnp.diff(result.values) <= 1e-10)
    assert result.grad_norms[-1] < result.grad_norms[0]
    assert jnp.min(jnp.linalg.eigvalsh(result.sigma)) > 0.0
