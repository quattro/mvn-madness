"""Energy and differential checks for the Kobayashi gauge fiber."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gaussian_fisher import _config as _config
from gaussian_fisher.kobayashi import gauge_matrix, horizontal_block, log_gauge_matrix
from gaussian_fisher.linalg import frechet_log_spd, skew, skew_from_vector, vector_from_skew


def _omega_from_input(omega: jnp.ndarray, dimension: int) -> jnp.ndarray:
    if omega.ndim == 1:
        return skew_from_vector(omega, dimension)
    return skew(omega)


def energy(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """Compute ``0.5 * ||log(G_omega)||_F^2``."""

    omega = _omega_from_input(omega, mu.shape[0])
    log_g = log_gauge_matrix(mu, sigma, omega)
    return 0.5 * jnp.sum(log_g * log_g)


def explicit_energy_gradient_matrix(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """Gradient under ``<A, B>_so = -tr(A B)``.

    The Kobayashi horizontal condition gives ``grad E = -2 * (log G)_{13}``.
    The skew projection removes roundoff-level symmetric leakage from the block.
    """

    omega = _omega_from_input(omega, mu.shape[0])
    block = horizontal_block(log_gauge_matrix(mu, sigma, omega), mu.shape[0])
    return skew(-2.0 * block)


def explicit_energy_gradient(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    return explicit_energy_gradient_matrix(mu, sigma, omega)


def energy_from_skew_vector(params: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    return energy(mu, sigma, params)


def explicit_gradient(mu: jnp.ndarray, sigma: jnp.ndarray, omega_vec: jnp.ndarray) -> jnp.ndarray:
    """Return the coordinate gradient matching ``jax.grad`` on skew coordinates."""

    matrix_gradient = explicit_energy_gradient_matrix(mu, sigma, omega_vec)
    return 2.0 * vector_from_skew(matrix_gradient)


def explicit_gradient_vector(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    return explicit_gradient(mu, sigma, vector_from_skew(_omega_from_input(omega, mu.shape[0])))


def autodiff_gradient(mu: jnp.ndarray, sigma: jnp.ndarray, omega_vec: jnp.ndarray) -> jnp.ndarray:
    return jax.grad(energy_from_skew_vector)(omega_vec, mu, sigma)


def autodiff_gradient_vector(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    return autodiff_gradient(mu, sigma, vector_from_skew(_omega_from_input(omega, mu.shape[0])))


def horizontal_residual(mu: jnp.ndarray, sigma: jnp.ndarray, omega_vec: jnp.ndarray) -> jnp.ndarray:
    omega = _omega_from_input(omega_vec, mu.shape[0])
    block = horizontal_block(log_gauge_matrix(mu, sigma, omega), mu.shape[0])
    return vector_from_skew(skew(block))


def _gauge_tangent_from_skew(mu: jnp.ndarray, sigma: jnp.ndarray, omega: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
    dimension = mu.shape[0]
    zeros_dd = jnp.zeros((dimension, dimension), dtype=mu.dtype)
    zeros_d1 = jnp.zeros((dimension, 1), dtype=mu.dtype)
    zeros_1d = jnp.zeros((1, dimension), dtype=mu.dtype)
    n_xi = jnp.block(
        [
            [zeros_dd, zeros_d1, zeros_dd],
            [zeros_1d, jnp.zeros((1, 1), dtype=mu.dtype), zeros_1d],
            [xi, zeros_d1, zeros_dd],
        ]
    )
    gauge = gauge_matrix(mu, sigma, omega)
    return n_xi @ gauge + gauge @ n_xi.T


def hessian_vector_explicit(mu: jnp.ndarray, sigma: jnp.ndarray, omega_vec: jnp.ndarray, xi_vec: jnp.ndarray) -> jnp.ndarray:
    omega = skew_from_vector(omega_vec, mu.shape[0])
    xi = skew_from_vector(xi_vec, mu.shape[0])
    tangent = _gauge_tangent_from_skew(mu, sigma, omega, xi)
    dlog = frechet_log_spd(gauge_matrix(mu, sigma, omega), tangent)
    matrix_hvp = skew(-2.0 * horizontal_block(dlog, mu.shape[0]))
    return 2.0 * vector_from_skew(matrix_hvp)


def hessian_vector_autodiff(mu: jnp.ndarray, sigma: jnp.ndarray, omega_vec: jnp.ndarray, xi_vec: jnp.ndarray) -> jnp.ndarray:
    _, hvp = jax.jvp(lambda params: jax.grad(energy_from_skew_vector)(params, mu, sigma), (omega_vec,), (xi_vec,))
    return hvp


def hessian_vector_product(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    omega: jnp.ndarray,
    tangent: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the Hessian-vector product in skew-coordinate form."""

    params = vector_from_skew(_omega_from_input(omega, mu.shape[0]))
    tangent_params = vector_from_skew(_omega_from_input(tangent, mu.shape[0]))
    return skew_from_vector(hessian_vector_autodiff(mu, sigma, params, tangent_params), mu.shape[0])
