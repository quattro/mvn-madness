"""Riemannian gradient descent utilities for MVN Fisher-Rao geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from gaussian_fisher import _config as _config
from gaussian_fisher.expmap import expmap
from gaussian_fisher.linalg import inv_spd, symmetrize


@dataclass(frozen=True)
class RGDResult:
    mu: jnp.ndarray
    sigma: jnp.ndarray
    values: jnp.ndarray
    step_sizes: jnp.ndarray
    grad_norms: jnp.ndarray
    success: bool
    status: str


Objective = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def euclidean_to_riemannian_gradient(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    grad_mu: jnp.ndarray,
    grad_sigma: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert Euclidean gradients to the Fisher-Rao Riemannian gradient."""

    del mu
    sigma = symmetrize(jnp.asarray(sigma))
    grad_sigma_sym = symmetrize(jnp.asarray(grad_sigma))
    v_mu = sigma @ jnp.asarray(grad_mu)
    v_sigma = 2.0 * sigma @ grad_sigma_sym @ sigma
    return v_mu, symmetrize(v_sigma)


def riemannian_gradient_direction(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    grad_mu: jnp.ndarray,
    grad_sigma: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the negative Fisher-Rao gradient direction for minimization."""

    v_mu, v_sigma = euclidean_to_riemannian_gradient(mu, sigma, grad_mu, grad_sigma)
    return -v_mu, -v_sigma


def fisher_tangent_norm_sq(
    sigma: jnp.ndarray,
    v_mu: jnp.ndarray,
    v_sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Squared Fisher norm of a tangent vector at covariance ``sigma``."""

    sigma_inv = inv_spd(sigma)
    mean_part = v_mu @ sigma_inv @ v_mu
    cov_part = 0.5 * jnp.trace(sigma_inv @ symmetrize(v_sigma) @ sigma_inv @ symmetrize(v_sigma))
    return mean_part + cov_part


def riemannian_gradient_step(
    objective: Objective,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    step_size: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute one Riemannian gradient descent step using the closed-form expmap."""

    value, grads = jax.value_and_grad(objective, argnums=(0, 1))(mu, sigma)
    grad_mu, grad_sigma = grads
    direction_mu, direction_sigma = riemannian_gradient_direction(mu, sigma, grad_mu, grad_sigma)
    grad_norm = jnp.sqrt(fisher_tangent_norm_sq(sigma, -direction_mu, -direction_sigma))
    next_mu, next_sigma = expmap(mu, sigma, direction_mu, direction_sigma, t=step_size)
    return next_mu, next_sigma, value, grad_norm


def _accepted_step(
    objective: Objective,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    direction_mu: jnp.ndarray,
    direction_sigma: jnp.ndarray,
    value: jnp.ndarray,
    initial_step_size: float,
    *,
    backtracking: bool,
    min_step_size: float,
) -> tuple[jnp.ndarray, jnp.ndarray, float, bool]:
    step = float(initial_step_size)
    while True:
        candidate_mu, candidate_sigma = expmap(mu, sigma, direction_mu, direction_sigma, t=step)
        candidate_value = objective(candidate_mu, candidate_sigma)
        accepted = bool(jnp.isfinite(candidate_value) & (candidate_value <= value))
        if accepted or not backtracking or step <= min_step_size:
            return candidate_mu, candidate_sigma, step, accepted
        step *= 0.5


def rgd(
    objective: Objective,
    mu0: jnp.ndarray,
    sigma0: jnp.ndarray,
    step_size: float,
    n_steps: int,
    backtracking: bool = True,
    min_step_size: float = 1e-8,
) -> RGDResult:
    """Basic host-side Riemannian gradient descent using the Fisher-Rao expmap."""

    mu = jnp.asarray(mu0)
    sigma = symmetrize(jnp.asarray(sigma0))
    values = []
    step_sizes = []
    grad_norms = []
    success = True
    status = "ok"

    for _ in range(n_steps):
        value, grads = jax.value_and_grad(objective, argnums=(0, 1))(mu, sigma)
        grad_mu, grad_sigma = grads
        direction_mu, direction_sigma = riemannian_gradient_direction(mu, sigma, grad_mu, grad_sigma)
        grad_norm = jnp.sqrt(fisher_tangent_norm_sq(sigma, -direction_mu, -direction_sigma))
        next_mu, next_sigma, accepted_step, accepted = _accepted_step(
            objective,
            mu,
            sigma,
            direction_mu,
            direction_sigma,
            value,
            step_size,
            backtracking=backtracking,
            min_step_size=min_step_size,
        )
        values.append(value)
        step_sizes.append(accepted_step)
        grad_norms.append(grad_norm)
        if not accepted:
            success = False
            status = "line_search_failed"
            break
        mu, sigma = next_mu, next_sigma

    values.append(objective(mu, sigma))
    return RGDResult(
        mu,
        sigma,
        jnp.asarray(values),
        jnp.asarray(step_sizes),
        jnp.asarray(grad_norms),
        success,
        status,
    )
