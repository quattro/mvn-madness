"""Diffrax shooting comparison from horizontal-lift initial velocities."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_fisher.geodesic import projected_geodesic_path
from gaussian_fisher.ode import integrate_geodesic_ode
from gaussian_fisher.solvers import minimize_gauge


def run_case(seed: int, dimension: int, h: float) -> dict[str, float | int | str]:
    key = jr.PRNGKey(seed)
    raw = jr.normal(key, (dimension, dimension)) * 0.15
    sigma = raw @ raw.T + jnp.eye(dimension)
    mu = jr.normal(key, (dimension,)) * 0.15
    result = minimize_gauge(mu, sigma, max_steps=512, tolerance=1e-10)
    short_path = projected_geodesic_path(mu, sigma, jnp.array([0.0, h]), omega=result.omega_star)
    mu_dot0 = (short_path.mu[1] - short_path.mu[0]) / h
    sigma_dot0 = (short_path.sigma[1] - short_path.sigma[0]) / h
    solution = integrate_geodesic_ode(
        (jnp.zeros(dimension), jnp.eye(dimension), mu_dot0, sigma_dot0),
        jnp.linspace(0.0, 1.0, 51),
        dt0=1e-3,
    )
    mus, sigmas, _, _ = solution.ys
    return {
        "dimension": dimension,
        "condition": float(jnp.linalg.cond(sigma)),
        "mu_norm": float(jnp.linalg.norm(mu)),
        "energy": float(result.energy),
        "horizontal_residual": float(result.horizontal_norm),
        "mu_terminal_error": float(jnp.linalg.norm(mus[-1] - mu)),
        "sigma_terminal_error": float(jnp.linalg.norm(sigmas[-1] - sigma)),
        "gradient_norm": float(result.gradient_norm),
        "solver_status": str(result.solver_result),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--h", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("experiments/output/shooting_with_diffrax.csv"))
    args = parser.parse_args()
    rows = [run_case(args.seed + dimension, dimension, args.h) for dimension in args.dimensions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

