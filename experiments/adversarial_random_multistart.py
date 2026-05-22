"""Random adversarial multistart scans for d=3,4,5 endpoints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from adversarial_common import (
    bool_int,
    endpoint_error_and_ode_residual,
    hessian_min_eigenvalue,
    precision_projection_diagnostics,
    random_mean_with_norm,
    random_spd_with_condition,
    write_rows,
)

from gaussian_fisher.solvers import minimize_gauge


def scan_endpoint(seed: int, dimension: int, endpoint_index: int, starts: int, *, max_steps: int) -> list[dict[str, Any]]:
    key = jr.PRNGKey(seed)
    k_cond, k_sigma, k_mean_norm, k_mu, k_starts = jr.split(key, 5)
    condition = float(10.0 ** jr.uniform(k_cond, (), minval=0.0, maxval=6.0))
    mean_norm = float(10.0 ** jr.uniform(k_mean_norm, (), minval=-3.0, maxval=2.0))
    sigma = random_spd_with_condition(k_sigma, dimension, condition)
    mu = random_mean_with_norm(k_mu, dimension, mean_norm)
    size = dimension * (dimension - 1) // 2
    start_keys = jr.split(k_starts, starts)
    rows: list[dict[str, Any]] = []
    returned_omegas = []
    returned_energies = []
    for start_index, start_key in enumerate(start_keys):
        omega0 = jr.normal(start_key, (size,)) * 5.0
        try:
            result = minimize_gauge(mu, sigma, omega0, max_steps=max_steps, tolerance=1e-9)
            endpoint_error, ode_residual = endpoint_error_and_ode_residual(mu, sigma, result.omega_star, samples=31)
            precision_diagnostics = precision_projection_diagnostics(mu, sigma, result.omega_star)
            hessian_min = hessian_min_eigenvalue(mu, sigma, result.omega_vec)
            returned_omegas.append(result.omega_vec)
            returned_energies.append(float(result.energy))
            rows.append(
                {
                    "scan_type": "random_multistart",
                    "dimension": dimension,
                    "endpoint": endpoint_index,
                    "start": start_index,
                    "seed": seed,
                    "condition": float(jnp.linalg.cond(sigma)),
                    "target_condition": condition,
                    "mean_norm": float(jnp.linalg.norm(mu)),
                    "energy": float(result.energy),
                    "horizontal_residual": float(result.horizontal_norm),
                    "endpoint_error": endpoint_error,
                    "condition_sigma_target": precision_diagnostics["condition_sigma_target"],
                    "condition_theta_projected": precision_diagnostics["condition_theta_projected"],
                    "condition_gauge": precision_diagnostics["condition_gauge"],
                    "lifted_endpoint_error": precision_diagnostics["lifted_endpoint_error"],
                    "relative_theta_error": precision_diagnostics["relative_theta_error"],
                    "relative_delta_error": precision_diagnostics["relative_delta_error"],
                    "precision_endpoint_error": precision_diagnostics["precision_endpoint_error"],
                    "relative_sigma_error": precision_diagnostics["relative_sigma_error"],
                    "relative_mu_error": precision_diagnostics["relative_mu_error"],
                    "covariance_endpoint_error": precision_diagnostics["covariance_endpoint_error"],
                    "projection_failure_class": precision_diagnostics["projection_failure_class"],
                    "ode_residual": ode_residual,
                    "gradient_norm": float(result.gradient_norm),
                    "hessian_min_eigenvalue": hessian_min,
                    "success": int(result.success_flag),
                    "omega_norm": float(jnp.linalg.norm(result.omega_vec)),
                    "solver_status": str(result.solver_result),
                    "energy_spread": 0.0,
                    "omega_spread": 0.0,
                    "multiple_minima": 0,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "scan_type": "random_multistart",
                    "dimension": dimension,
                    "endpoint": endpoint_index,
                    "start": start_index,
                    "seed": seed,
                    "condition": float(jnp.linalg.cond(sigma)),
                    "target_condition": condition,
                    "mean_norm": float(jnp.linalg.norm(mu)),
                    "energy": math.nan,
                    "horizontal_residual": math.nan,
                    "endpoint_error": math.nan,
                    "condition_sigma_target": float(jnp.linalg.cond(sigma)),
                    "condition_theta_projected": math.nan,
                    "condition_gauge": math.nan,
                    "lifted_endpoint_error": math.nan,
                    "relative_theta_error": math.nan,
                    "relative_delta_error": math.nan,
                    "precision_endpoint_error": math.nan,
                    "relative_sigma_error": math.nan,
                    "relative_mu_error": math.nan,
                    "covariance_endpoint_error": math.nan,
                    "projection_failure_class": "code_error",
                    "ode_residual": math.nan,
                    "gradient_norm": math.nan,
                    "hessian_min_eigenvalue": math.nan,
                    "success": 0,
                    "omega_norm": math.nan,
                    "solver_status": f"code_error:{type(exc).__name__}:{exc}",
                    "energy_spread": math.nan,
                    "omega_spread": math.nan,
                    "multiple_minima": 0,
                }
            )
    if returned_omegas:
        omega_stack = jnp.stack(returned_omegas)
        energy_spread = max(returned_energies) - min(returned_energies)
        omega_spread = float(jnp.max(jnp.linalg.norm(omega_stack - omega_stack[0], axis=1)))
        multiple_minima = bool_int(energy_spread > 1e-7 or omega_spread > 1e-5)
    else:
        energy_spread = math.nan
        omega_spread = math.nan
        multiple_minima = 0
    for row in rows:
        row["energy_spread"] = energy_spread
        row["omega_spread"] = omega_spread
        row["multiple_minima"] = multiple_minima
    return rows


def build_rows(seed: int, dimensions: tuple[int, ...], endpoints: int, starts: int, max_steps: int) -> list[dict[str, Any]]:
    rows = []
    case_seed = seed
    for dimension in dimensions:
        for endpoint_index in range(endpoints):
            rows.extend(scan_endpoint(case_seed, dimension, endpoint_index, starts, max_steps=max_steps))
            case_seed += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--endpoints", type=int, default=10)
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--output", type=Path, default=Path("experiments/output/adversarial_random_multistart.csv"))
    args = parser.parse_args()
    rows = build_rows(args.seed, tuple(args.dimensions), args.endpoints, args.starts, args.max_steps)
    write_rows(args.output, rows)
    print(f"wrote {args.output}")
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()
