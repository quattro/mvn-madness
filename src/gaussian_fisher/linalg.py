"""JAX linear algebra helpers for symmetric positive definite matrices."""

from __future__ import annotations

from gaussian_fisher import _config as _config
import jax.numpy as jnp


def symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return the symmetric part of a square matrix."""

    return 0.5 * (matrix + matrix.T)


def sym(matrix: jnp.ndarray) -> jnp.ndarray:
    return symmetrize(matrix)


def skew(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return the skew-symmetric part of a square matrix."""

    return 0.5 * (matrix - matrix.T)


def check_square_matrix(matrix: jnp.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")


def eigh_spd(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Eigen-decompose a symmetric positive definite matrix."""

    check_square_matrix(matrix, "matrix")
    values, vectors = jnp.linalg.eigh(symmetrize(matrix))
    return values, vectors


def fractional_power_spd(matrix: jnp.ndarray, power: float) -> jnp.ndarray:
    """Compute an SPD matrix power through a symmetric eigendecomposition."""

    values, vectors = eigh_spd(matrix)
    return (vectors * (values**power)) @ vectors.T


def matrix_power_spd(matrix: jnp.ndarray, power: float) -> jnp.ndarray:
    return fractional_power_spd(matrix, power)


def sqrtm_spd(matrix: jnp.ndarray) -> jnp.ndarray:
    return fractional_power_spd(matrix, 0.5)


def invsqrtm_spd(matrix: jnp.ndarray) -> jnp.ndarray:
    return fractional_power_spd(matrix, -0.5)


def logm_spd(matrix: jnp.ndarray) -> jnp.ndarray:
    values, vectors = eigh_spd(matrix)
    return (vectors * jnp.log(values)) @ vectors.T


def expm_sym(matrix: jnp.ndarray) -> jnp.ndarray:
    values, vectors = jnp.linalg.eigh(symmetrize(matrix))
    return (vectors * jnp.exp(values)) @ vectors.T


def inv_spd(matrix: jnp.ndarray) -> jnp.ndarray:
    values, vectors = eigh_spd(matrix)
    return (vectors * (1.0 / values)) @ vectors.T


def is_spd(matrix: jnp.ndarray, *, tol: float = 1e-8) -> bool:
    values = jnp.linalg.eigvalsh(symmetrize(matrix))
    return bool(jnp.all(values > tol))


def skew_from_vector(params: jnp.ndarray, dimension: int) -> jnp.ndarray:
    expected = dimension * (dimension - 1) // 2
    if params.shape != (expected,):
        raise ValueError(f"params must have shape ({expected},)")
    out = jnp.zeros((dimension, dimension), dtype=params.dtype)
    idx = jnp.triu_indices(dimension, k=1)
    out = out.at[idx].set(params)
    return out - out.T


def make_skew_from_vec(params: jnp.ndarray, dimension: int) -> jnp.ndarray:
    return skew_from_vector(params, dimension)


def vector_from_skew(matrix: jnp.ndarray) -> jnp.ndarray:
    check_square_matrix(matrix, "matrix")
    idx = jnp.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def skew_to_vec(matrix: jnp.ndarray) -> jnp.ndarray:
    return vector_from_skew(matrix)


def frechet_log_spd(matrix: jnp.ndarray, tangent: jnp.ndarray, *, equal_tol: float = 1e-10) -> jnp.ndarray:
    """Frechet derivative of the matrix logarithm at an SPD matrix.

    Uses divided differences in the eigenbasis:
    ``D log_G[H] = U (log^[1](lambda_i, lambda_j) * (U.T H U)_ij) U.T``.
    """

    values, vectors = eigh_spd(matrix)
    tangent_hat = vectors.T @ symmetrize(tangent) @ vectors
    left = values[:, None]
    right = values[None, :]
    separated = jnp.abs(left - right) > equal_tol * jnp.maximum(1.0, jnp.maximum(jnp.abs(left), jnp.abs(right)))
    divided = jnp.where(separated, (jnp.log(left) - jnp.log(right)) / (left - right), 1.0 / (0.5 * (left + right)))
    return symmetrize(vectors @ (divided * tangent_hat) @ vectors.T)
