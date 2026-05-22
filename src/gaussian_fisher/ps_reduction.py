"""Scalar covariance reduction to the Poincare half-plane."""

from __future__ import annotations

import jax.numpy as jnp

from gaussian_fisher import _config as _config

def gaussian_to_half_plane(mu: jnp.ndarray, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map one-dimensional Gaussian coordinates to half-plane coordinates.

    The Fisher metric becomes ``2 * (dx^2 + dy^2) / y^2`` under
    ``x = mu / sqrt(2)``, ``y = sqrt(sigma)``.
    """

    return mu / jnp.sqrt(2.0), jnp.sqrt(sigma)


def half_plane_to_gaussian(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.sqrt(2.0) * x, y * y


def hyperbolic_distance(x0: jnp.ndarray, y0: jnp.ndarray, x1: jnp.ndarray, y1: jnp.ndarray) -> jnp.ndarray:
    cosh_distance = 1.0 + ((x1 - x0) ** 2 + (y1 - y0) ** 2) / (2.0 * y0 * y1)
    return jnp.arccosh(cosh_distance)


def scalar_fisher_distance(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray) -> jnp.ndarray:
    x0, y0 = gaussian_to_half_plane(mu0, sigma0)
    x1, y1 = gaussian_to_half_plane(mu1, sigma1)
    return jnp.sqrt(2.0) * hyperbolic_distance(x0, y0, x1, y1)


def solve_ps_reduction(*_args, **_kwargs):
    """Experimental placeholder for the speculative P,S reduction.

    The candidate equations are intentionally not trusted yet. Keep this
    explicit until tests demonstrate agreement with variational minimization.
    """

    raise NotImplementedError("P,S reduction is speculative and not yet validated")
