"""Kobayashi-style gauge lift for normalized Gaussian endpoints."""

from __future__ import annotations

import jax.numpy as jnp

from gaussian_fisher import _config as _config
from gaussian_fisher.linalg import inv_spd, logm_spd, skew, symmetrize


def lift_matrix(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """Build the lower triangular lift matrix ``L_omega``."""

    dimension = mu.shape[0]
    top = jnp.concatenate([jnp.eye(dimension), jnp.zeros((dimension, 1)), jnp.zeros((dimension, dimension))], axis=1)
    middle = jnp.concatenate([mu[None, :], jnp.ones((1, 1)), jnp.zeros((1, dimension))], axis=1)
    lower_left = omega - 0.5 * jnp.outer(mu, mu)
    bottom = jnp.concatenate([lower_left, -mu[:, None], jnp.eye(dimension)], axis=1)
    return jnp.concatenate([top, middle, bottom], axis=0)


def build_L(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    return lift_matrix(mu, sigma, omega)


def diagonal_metric_matrix(sigma: jnp.ndarray) -> jnp.ndarray:
    """Build ``D_theta`` with ``theta = sigma^{-1}``."""

    dimension = sigma.shape[0]
    theta = inv_spd(sigma)
    return jnp.block(
        [
            [theta, jnp.zeros((dimension, 1)), jnp.zeros((dimension, dimension))],
            [jnp.zeros((1, dimension)), jnp.ones((1, 1)), jnp.zeros((1, dimension))],
            [jnp.zeros((dimension, dimension)), jnp.zeros((dimension, 1)), sigma],
        ]
    )


def build_D(sigma: jnp.ndarray) -> jnp.ndarray:
    return diagonal_metric_matrix(sigma)


def gauge_matrix(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """Return ``G_omega = L_omega D_theta L_omega.T``."""

    lift = lift_matrix(mu, sigma, omega)
    return symmetrize(lift @ diagonal_metric_matrix(sigma) @ lift.T)


def build_G(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    return gauge_matrix(mu, sigma, omega)


def log_gauge_matrix(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    return logm_spd(gauge_matrix(mu, sigma, omega))


def extract_blocks(matrix: jnp.ndarray, dimension: int) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    i0 = dimension
    i1 = dimension + 1
    return (
        (matrix[:i0, :i0], matrix[:i0, i0:i1], matrix[:i0, i1:]),
        (matrix[i0:i1, :i0], matrix[i0:i1, i0:i1], matrix[i0:i1, i1:]),
        (matrix[i1:, :i0], matrix[i1:, i0:i1], matrix[i1:, i1:]),
    )


def horizontal_block(log_gauge: jnp.ndarray, dimension: int) -> jnp.ndarray:
    """Extract the top-right ``(1, 3)`` block in the ``(d, 1, d)`` partition."""

    return log_gauge[:dimension, dimension + 1 :]


def horizontal_residual(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    block = horizontal_block(log_gauge_matrix(mu, sigma, omega), mu.shape[0])
    return skew(block)
