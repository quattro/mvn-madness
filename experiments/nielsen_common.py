"""Shared helpers for Nielsen approximation experiments."""

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

from gaussian_fisher.geodesic import projected_precision_geodesic_path
from gaussian_fisher.hilbert_proxy import hilbert_spd_distance
from gaussian_fisher.linalg import invsqrtm_spd, logm_spd
from gaussian_fisher.normalize import normalize_endpoint
from gaussian_fisher.ode_checks import precision_residual_summary
from gaussian_fisher.ps_reduction import scalar_fisher_distance
from gaussian_fisher.solvers import minimize_gauge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from adversarial_common import precision_projection_diagnostics


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty row set")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def random_spd(key: jax.Array, dimension: int, scale: float = 0.35) -> jnp.ndarray:
    raw = jr.normal(key, (dimension, dimension)) * scale
    return raw @ raw.T + jnp.eye(dimension)


def horizontal_lift_distance(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray) -> dict[str, Any]:
    endpoint = normalize_endpoint(mu0, sigma0, mu1, sigma1)
    result = minimize_gauge(endpoint.mu, endpoint.sigma, max_steps=512, tolerance=1e-10)
    path = projected_precision_geodesic_path(endpoint.mu, endpoint.sigma, jnp.linspace(0.0, 1.0, 51), omega=result.omega_star)
    residual = precision_residual_summary(path.delta, path.theta, path.t)
    precision = precision_projection_diagnostics(endpoint.mu, endpoint.sigma, result.omega_star)
    return {
        "distance": float(jnp.sqrt(0.5 * result.energy)),
        "energy": float(result.energy),
        "horizontal_residual": float(result.horizontal_norm),
        "gradient_norm": float(result.gradient_norm),
        "solver_status": str(result.solver_result),
        "ode_residual": float(jnp.maximum(residual.max_delta_abs, residual.max_theta_abs)),
        "ode_residual_coordinates": "precision",
        "precision_ode_delta_abs": float(residual.max_delta_abs),
        "precision_ode_theta_abs": float(residual.max_theta_abs),
        "precision_ode_delta_rel": float(residual.max_delta_rel),
        "precision_ode_theta_rel": float(residual.max_theta_rel),
        "lifted_endpoint_error": precision["lifted_endpoint_error"],
        "precision_endpoint_error": precision["precision_endpoint_error"],
        "covariance_endpoint_error": precision["covariance_endpoint_error"],
        "projection_failure_class": precision["projection_failure_class"],
    }


def exact_distance_if_available(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray) -> float:
    if mu0.shape[0] == 1:
        return float(scalar_fisher_distance(mu0[0], sigma0[0, 0], mu1[0], sigma1[0, 0]))
    if bool(jnp.allclose(mu0, mu1, atol=1e-12)):
        normalized = invsqrtm_spd(sigma0) @ sigma1 @ invsqrtm_spd(sigma0)
        return float(jnp.linalg.norm(logm_spd(normalized)) / jnp.sqrt(2.0))
    return math.nan


def condition_and_mean_norm(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray) -> tuple[float, float]:
    endpoint = normalize_endpoint(mu0, sigma0, mu1, sigma1)
    return float(jnp.linalg.cond(endpoint.sigma)), float(jnp.linalg.norm(endpoint.mu))


def hilbert_covariance_proxy(sigma0: jnp.ndarray, sigma1: jnp.ndarray) -> float:
    return float(hilbert_spd_distance(sigma0, sigma1))
