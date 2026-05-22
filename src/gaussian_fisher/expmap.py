"""Closed-form Fisher-Rao exponential map for multivariate normals."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import lineax as lx

from gaussian_fisher import _config as _config
from gaussian_fisher.linalg import expm_sym, invsqrtm_spd, sqrtm_spd, symmetrize


class NormalizedTangent(NamedTuple):
    x: jnp.ndarray
    b: jnp.ndarray
    sigma_sqrt: jnp.ndarray
    sigma_invsqrt: jnp.ndarray


def _solve_spd(matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    operator = lx.MatrixLinearOperator(matrix, tags=lx.positive_semidefinite_tag)
    solver = lx.AutoLinearSolver(well_posed=True)
    solve_vector = lambda vector: lx.linear_solve(operator, vector, solver=solver).value
    if rhs.ndim == 1:
        return solve_vector(rhs)
    return jax.vmap(solve_vector, in_axes=1, out_axes=1)(rhs)


def _spectral_sinh_over_g(values: jnp.ndarray, t: jnp.ndarray, *, tol: float = 1e-12) -> jnp.ndarray:
    z = t * values
    small = jnp.abs(values) <= tol
    return jnp.where(small, t + (t**3) * (values**2) / 6.0, jnp.sinh(z) / values)


def _spectral_cosh_minus_one_over_g2(values: jnp.ndarray, t: jnp.ndarray, *, tol: float = 1e-8) -> jnp.ndarray:
    z = t * values
    small = jnp.abs(values) <= tol
    return jnp.where(
        small,
        0.5 * t**2 + (t**4) * (values**2) / 24.0,
        0.5 * (jnp.expm1(z) + jnp.expm1(-z)) / (values**2),
    )


def _spectral_apply(vectors: jnp.ndarray, diagonal: jnp.ndarray) -> jnp.ndarray:
    return symmetrize((vectors * diagonal) @ vectors.T)


def _closed_form_precision_terms(x: jnp.ndarray, b: jnp.ndarray, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    d = x.shape[0]
    g2 = symmetrize(b @ b + 2.0 * jnp.outer(x, x))
    values2, vectors = jnp.linalg.eigh(g2)
    values = jnp.sqrt(jnp.maximum(values2, 0.0))

    cosh_g = _spectral_apply(vectors, jnp.cosh(t * values))
    sinh_over_g = _spectral_apply(vectors, _spectral_sinh_over_g(values, t))
    cosh_minus_one_over_g2 = _spectral_apply(vectors, _spectral_cosh_minus_one_over_g2(values, t))

    identity = jnp.eye(d, dtype=x.dtype)
    c_minus_i = cosh_g - identity
    delta_precision = (
        identity
        + 0.5 * c_minus_i
        + 0.5 * b @ cosh_minus_one_over_g2 @ b
        - 0.5 * sinh_over_g @ b
        - 0.5 * b @ sinh_over_g
    )
    delta = -b @ cosh_minus_one_over_g2 @ x + sinh_over_g @ x
    return delta, symmetrize(delta_precision)


def normalize_tangent(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    v_mu: jnp.ndarray,
    v_sigma: jnp.ndarray,
) -> NormalizedTangent:
    """Normalize a tangent at ``(mu, Sigma)`` to one at ``(0, I)``.

    Returns ``(x, B, A, A_inv)`` where ``A = Sigma^{1/2}``,
    ``x = A_inv @ v_mu`` and ``B = A_inv @ v_Sigma @ A_inv``.
    """

    del mu
    sigma = symmetrize(jnp.asarray(sigma))
    sigma_sqrt = sqrtm_spd(sigma)
    sigma_invsqrt = invsqrtm_spd(sigma)
    x = sigma_invsqrt @ jnp.asarray(v_mu)
    b = sigma_invsqrt @ symmetrize(jnp.asarray(v_sigma)) @ sigma_invsqrt
    return NormalizedTangent(x, symmetrize(b), sigma_sqrt, sigma_invsqrt)


def expmap_normalized(x: jnp.ndarray, b: jnp.ndarray, t: float | jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Closed-form Fisher-Rao exponential map from ``(0, I)``."""

    x = jnp.asarray(x)
    b = symmetrize(jnp.asarray(b))
    time = jnp.asarray(t, dtype=x.dtype)
    delta, precision = _closed_form_precision_terms(x, b, time)
    mu_tilde = _solve_spd(precision, delta)
    sigma_tilde = _solve_spd(precision, jnp.eye(x.shape[0], dtype=x.dtype))
    return mu_tilde, symmetrize(sigma_tilde)


def expmap(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    v_mu: jnp.ndarray,
    v_sigma: jnp.ndarray,
    t: float | jnp.ndarray = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Closed-form Fisher-Rao exponential map from an arbitrary base point."""

    tangent = normalize_tangent(mu, sigma, v_mu, v_sigma)
    mu_tilde, sigma_tilde = expmap_normalized(tangent.x, tangent.b, t)
    mu_t = jnp.asarray(mu) + tangent.sigma_sqrt @ mu_tilde
    sigma_t = tangent.sigma_sqrt @ sigma_tilde @ tangent.sigma_sqrt
    return mu_t, symmetrize(sigma_t)


def expmap_path(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    v_mu: jnp.ndarray,
    v_sigma: jnp.ndarray,
    ts: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized exponential-map path over times ``ts``."""

    tangent = normalize_tangent(mu, sigma, v_mu, v_sigma)

    def one_time(time: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        mu_tilde, sigma_tilde = expmap_normalized(tangent.x, tangent.b, time)
        mu_t = jnp.asarray(mu) + tangent.sigma_sqrt @ mu_tilde
        sigma_t = tangent.sigma_sqrt @ sigma_tilde @ tangent.sigma_sqrt
        return mu_t, symmetrize(sigma_t)

    return jax.vmap(one_time)(jnp.asarray(ts))


def covariance_only_expmap(mu: jnp.ndarray, sigma: jnp.ndarray, v_sigma: jnp.ndarray, t: float | jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reference covariance-only geodesic used by tests and examples."""

    sigma_sqrt = sqrtm_spd(sigma)
    sigma_invsqrt = invsqrtm_spd(sigma)
    normalized_tangent = sigma_invsqrt @ symmetrize(v_sigma) @ sigma_invsqrt
    sigma_t = sigma_sqrt @ expm_sym(jnp.asarray(t) * normalized_tangent) @ sigma_sqrt
    return jnp.asarray(mu), symmetrize(sigma_t)
