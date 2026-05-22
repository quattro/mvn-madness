"""Affine endpoint normalization for Gaussian Fisher-Rao problems."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from gaussian_fisher import _config as _config
from gaussian_fisher.linalg import invsqrtm_spd, sqrtm_spd, symmetrize


class NormalizedEndpoint(NamedTuple):
    mu: jnp.ndarray
    sigma: jnp.ndarray
    mu0: jnp.ndarray
    sigma0: jnp.ndarray
    sigma0_sqrt: jnp.ndarray
    sigma0_invsqrt: jnp.ndarray


def normalize_endpoint(
    mu0: jnp.ndarray,
    sigma0: jnp.ndarray,
    mu1: jnp.ndarray,
    sigma1: jnp.ndarray,
) -> NormalizedEndpoint:
    """Normalize endpoints from ``(mu0, sigma0) -> (mu1, sigma1)`` to ``(0, I) -> (mu, sigma)``."""

    mu0 = jnp.asarray(mu0)
    mu1 = jnp.asarray(mu1)
    sigma0 = jnp.asarray(sigma0)
    sigma1 = jnp.asarray(sigma1)
    if mu0.shape != mu1.shape:
        raise ValueError("mu0 and mu1 must have the same shape")
    if sigma0.shape != sigma1.shape or sigma0.shape != (mu0.shape[0], mu0.shape[0]):
        raise ValueError("covariance shapes must match endpoint dimension")

    sigma0_sqrt = sqrtm_spd(sigma0)
    sigma0_invsqrt = invsqrtm_spd(sigma0)
    normalized_mu = sigma0_invsqrt @ (mu1 - mu0)
    normalized_sigma = sigma0_invsqrt @ sigma1 @ sigma0_invsqrt
    return NormalizedEndpoint(
        normalized_mu,
        symmetrize(normalized_sigma),
        mu0,
        sigma0,
        sigma0_sqrt,
        sigma0_invsqrt,
    )


def denormalize_point(endpoint: NormalizedEndpoint, mu: jnp.ndarray, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map one normalized point back to the original affine coordinates."""

    return endpoint.mu0 + endpoint.sigma0_sqrt @ mu, symmetrize(endpoint.sigma0_sqrt @ sigma @ endpoint.sigma0_sqrt)


def denormalize_path(
    endpoint: NormalizedEndpoint,
    mus: jnp.ndarray,
    sigmas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map a batch of normalized path points back to original affine coordinates."""

    out_mu = endpoint.mu0 + jnp.einsum("ij,tj->ti", endpoint.sigma0_sqrt, mus)
    out_sigma = jnp.einsum("ij,tjk,lk->til", endpoint.sigma0_sqrt, sigmas, endpoint.sigma0_sqrt)
    return out_mu, jnp.vectorize(symmetrize, signature="(m,m)->(m,m)")(out_sigma)


def unnormalize_path(
    mu0: jnp.ndarray,
    sigma0: jnp.ndarray,
    mu_path: jnp.ndarray,
    sigma_path: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    endpoint = normalize_endpoint(mu0, sigma0, mu0, sigma0)
    return denormalize_path(endpoint, mu_path, sigma_path)
