"""Heuristic random search for small Hessian eigenvalues at gauge minimizers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from adversarial_common import endpoint_error_and_ode_residual, hessian_min_eigenvalue, random_mean_with_norm, random_spd_with_condition, write_rows

from gaussian_fisher.solvers import minimize_gauge


def run_search(seed: int, dimensions: tuple[int, ...], samples: int, keep: int, max_steps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    key = jr.PRNGKey(seed)
    for sample in range(samples):
        key, k_dim, k_cond, k_sigma, k_mean_norm, k_mu, k_start = jr.split(key, 7)
        dimension = int(jnp.asarray(dimensions)[jr.randint(k_dim, (), 0, len(dimensions))])
        condition = float(10.0 ** jr.uniform(k_cond, (), minval=0.0, maxval=6.0))
        mean_norm = float(10.0 ** jr.uniform(k_mean_norm, (), minval=-3.0, maxval=2.0))
        sigma = random_spd_with_condition(k_sigma, dimension, condition)
        mu = random_mean_with_norm(k_mu, dimension, mean_norm)
        omega0 = jr.normal(k_start, (dimension * (dimension - 1) // 2,)) * 2.0
        try:
            result = minimize_gauge(mu, sigma, omega0, max_steps=max_steps, tolerance=1e-9)
            hessian_min = hessian_min_eigenvalue(mu, sigma, result.omega_vec)
            endpoint_error, ode_residual = endpoint_error_and_ode_residual(mu, sigma, result.omega_star, samples=31)
            row = {
                "scan_type": "hessian_degeneracy",
                "dimension": dimension,
                "sample": sample,
                "seed": seed,
                "condition": float(jnp.linalg.cond(sigma)),
                "target_condition": condition,
                "mean_norm": float(jnp.linalg.norm(mu)),
                "energy": float(result.energy),
                "horizontal_residual": float(result.horizontal_norm),
                "endpoint_error": endpoint_error,
                "ode_residual": ode_residual,
                "gradient_norm": float(result.gradient_norm),
                "hessian_min_eigenvalue": hessian_min,
                "success": int(result.success_flag),
                "omega_norm": float(jnp.linalg.norm(result.omega_vec)),
                "solver_status": str(result.solver_result),
            }
        except Exception as exc:
            row = {
                "scan_type": "hessian_degeneracy",
                "dimension": dimension,
                "sample": sample,
                "seed": seed,
                "condition": float(jnp.linalg.cond(sigma)),
                "target_condition": condition,
                "mean_norm": float(jnp.linalg.norm(mu)),
                "energy": math.nan,
                "horizontal_residual": math.nan,
                "endpoint_error": math.nan,
                "ode_residual": math.nan,
                "gradient_norm": math.nan,
                "hessian_min_eigenvalue": math.nan,
                "success": 0,
                "omega_norm": math.nan,
                "solver_status": f"code_error:{type(exc).__name__}:{exc}",
            }
        rows.append(row)
    finite_rows = [row for row in rows if math.isfinite(float(row["hessian_min_eigenvalue"]))]
    finite_rows.sort(key=lambda row: float(row["hessian_min_eigenvalue"]))
    return finite_rows[:keep] + [row for row in rows if not math.isfinite(float(row["hessian_min_eigenvalue"]))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--keep", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--output", type=Path, default=Path("experiments/output/hessian_degeneracy_search.csv"))
    args = parser.parse_args()
    rows = run_search(args.seed, tuple(args.dimensions), args.samples, args.keep, args.max_steps)
    write_rows(args.output, rows)
    print(f"wrote {args.output}")
    print(f"rows {len(rows)}")
    if rows:
        print(f"smallest hessian eigenvalue {rows[0]['hessian_min_eigenvalue']}")


if __name__ == "__main__":
    main()

