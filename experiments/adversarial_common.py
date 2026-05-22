"""Shared utilities for adversarial gauge-energy experiments."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_fisher.energy import hessian_vector_autodiff
from gaussian_fisher.geodesic import lifted_geodesic, project_to_precision_gaussian, projected_geodesic_path, projected_precision_geodesic_path
from gaussian_fisher.kobayashi import gauge_matrix
from gaussian_fisher.linalg import inv_spd
from gaussian_fisher.ode_checks import precision_residual_summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty row set")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def random_orthogonal(key: jax.Array, dimension: int) -> jnp.ndarray:
    q, r = jnp.linalg.qr(jr.normal(key, (dimension, dimension)))
    signs = jnp.sign(jnp.diag(r))
    signs = jnp.where(signs == 0.0, 1.0, signs)
    return q * signs


def random_spd_with_condition(key: jax.Array, dimension: int, condition: float) -> jnp.ndarray:
    k_q, k_perm = jr.split(key)
    q = random_orthogonal(k_q, dimension)
    values = jnp.geomspace(1.0, condition, dimension)
    values = values[jr.permutation(k_perm, dimension)]
    return q @ jnp.diag(values) @ q.T


def random_mean_with_norm(key: jax.Array, dimension: int, norm: float) -> jnp.ndarray:
    direction = jr.normal(key, (dimension,))
    direction_norm = jnp.linalg.norm(direction)
    direction = jnp.where(direction_norm == 0.0, jnp.ones_like(direction), direction / direction_norm)
    return norm * direction


def hessian_matrix(mu: jnp.ndarray, sigma: jnp.ndarray, omega_vec: jnp.ndarray) -> jnp.ndarray:
    size = omega_vec.shape[0]
    if size == 0:
        return jnp.zeros((0, 0), dtype=omega_vec.dtype)
    basis = jnp.eye(size, dtype=omega_vec.dtype)
    return jnp.stack([hessian_vector_autodiff(mu, sigma, omega_vec, basis[i]) for i in range(size)], axis=1)


def hessian_min_eigenvalue(mu: jnp.ndarray, sigma: jnp.ndarray, omega_vec: jnp.ndarray) -> float:
    matrix = hessian_matrix(mu, sigma, omega_vec)
    if matrix.shape == (0, 0):
        return math.inf
    symmetric = 0.5 * (matrix + matrix.T)
    return float(jnp.min(jnp.linalg.eigvalsh(symmetric)))


def endpoint_error_and_ode_residual(mu: jnp.ndarray, sigma: jnp.ndarray, omega_star: jnp.ndarray, *, samples: int = 51) -> tuple[float, float]:
    path = projected_geodesic_path(mu, sigma, jnp.linspace(0.0, 1.0, samples), omega=omega_star)
    endpoint_error = jnp.linalg.norm(path.mu[-1] - mu) + jnp.linalg.norm(path.sigma[-1] - sigma)
    precision_path = projected_precision_geodesic_path(mu, sigma, path.t, omega=omega_star)
    residual = precision_residual_summary(precision_path.delta, precision_path.theta, precision_path.t)
    ode_residual = jnp.maximum(residual.max_delta_abs, residual.max_theta_abs)
    return float(endpoint_error), float(ode_residual)


def precision_projection_diagnostics(mu: jnp.ndarray, sigma: jnp.ndarray, omega_star: jnp.ndarray) -> dict[str, float]:
    """Compare lifted endpoint projection in precision and covariance coordinates."""

    target_gauge = gauge_matrix(mu, sigma, omega_star)
    reconstructed_gauge = lifted_geodesic(mu, sigma, omega_star, jnp.array([0.0, 1.0]))[-1]
    delta_projected, theta_projected = project_to_precision_gaussian(reconstructed_gauge, mu.shape[0])
    theta_target = inv_spd(sigma)
    delta_target = theta_target @ mu
    sigma_projected = inv_spd(theta_projected)
    mu_projected = sigma_projected @ delta_projected

    def relative_norm(error: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(error) / jnp.maximum(jnp.linalg.norm(reference), 1e-300)

    lifted_endpoint_error = jnp.linalg.norm(reconstructed_gauge - target_gauge) / jnp.maximum(jnp.linalg.norm(target_gauge), 1e-300)
    precision_endpoint_error = jnp.maximum(
        relative_norm(theta_projected - theta_target, theta_target),
        relative_norm(delta_projected - delta_target, delta_target),
    )
    covariance_endpoint_error = jnp.maximum(
        relative_norm(sigma_projected - sigma, sigma),
        relative_norm(mu_projected - mu, mu),
    )
    gauge_values = jnp.linalg.eigvalsh(target_gauge)
    if bool(lifted_endpoint_error < 1e-8 and precision_endpoint_error < 1e-8 and covariance_endpoint_error < 1e-6):
        failure_class = "success"
    elif bool(lifted_endpoint_error < 1e-8 and precision_endpoint_error < 1e-8 and covariance_endpoint_error >= 1e-6):
        failure_class = "projection_conditioning"
    else:
        failure_class = "geometric_or_solver_failure"
    return {
        "condition_sigma_target": float(jnp.linalg.cond(sigma)),
        "condition_theta_projected": float(jnp.linalg.cond(theta_projected)),
        "condition_gauge": float(jnp.max(gauge_values) / jnp.min(gauge_values)),
        "lifted_endpoint_error": float(lifted_endpoint_error),
        "relative_theta_error": float(relative_norm(theta_projected - theta_target, theta_target)),
        "relative_delta_error": float(relative_norm(delta_projected - delta_target, delta_target)),
        "precision_endpoint_error": float(precision_endpoint_error),
        "relative_sigma_error": float(relative_norm(sigma_projected - sigma, sigma)),
        "relative_mu_error": float(relative_norm(mu_projected - mu, mu)),
        "covariance_endpoint_error": float(covariance_endpoint_error),
        "projection_failure_class": failure_class,
    }


def unique_count(values: list[float], *, tolerance: float) -> int:
    unique: list[float] = []
    for value in sorted(values):
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    return len(unique)


def bool_int(value: bool) -> int:
    return 1 if value else 0
