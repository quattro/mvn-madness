"""Multi-start random endpoint scans for the Kobayashi gauge minimizer."""

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
from gaussian_fisher.ode_checks import residual_summary
from gaussian_fisher.solvers import minimize_gauge


def random_spd(key: jax.Array, dimension: int, scale: float) -> jnp.ndarray:
    raw = jr.normal(key, (dimension, dimension)) * scale
    return raw @ raw.T + jnp.eye(dimension)


def hessian_min_eigenvalue(mu: jnp.ndarray, sigma: jnp.ndarray, omega_vec: jnp.ndarray) -> jnp.ndarray:
    from gaussian_fisher.energy import hessian_vector_autodiff

    size = omega_vec.shape[0]
    if size == 0:
        return jnp.inf
    basis = jnp.eye(size)
    hessian = jnp.stack([hessian_vector_autodiff(mu, sigma, omega_vec, basis[i]) for i in range(size)], axis=1)
    return jnp.min(jnp.linalg.eigvalsh(0.5 * (hessian + hessian.T)))


def scan(seed: int, dimensions: tuple[int, ...], endpoints: int, starts: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    key = jr.PRNGKey(seed)
    for dimension in dimensions:
        for endpoint_index in range(endpoints):
            key, k_sigma, k_mu, k_start = jr.split(key, 4)
            sigma = random_spd(k_sigma, dimension, 0.25)
            mu = jr.normal(k_mu, (dimension,)) * 0.25
            start_keys = jr.split(k_start, starts)
            endpoint_results = []
            for start_index, start_key in enumerate(start_keys):
                omega0 = jr.normal(start_key, (dimension * (dimension - 1) // 2,)) * 0.5
                result = minimize_gauge(mu, sigma, omega0, max_steps=512, tolerance=1e-9)
                path = projected_geodesic_path(mu, sigma, jnp.linspace(0.0, 1.0, 51), omega=result.omega_star)
                residual = residual_summary(path.mu, path.sigma, path.t)
                endpoint_error = jnp.linalg.norm(path.mu[-1] - mu) + jnp.linalg.norm(path.sigma[-1] - sigma)
                hessian_min = hessian_min_eigenvalue(mu, sigma, result.omega_vec)
                endpoint_results.append(result)
                rows.append(
                    {
                        "dimension": dimension,
                        "endpoint": endpoint_index,
                        "start": start_index,
                        "condition": float(jnp.linalg.cond(sigma)),
                        "mu_norm": float(jnp.linalg.norm(mu)),
                        "energy": float(result.energy),
                        "horizontal_residual": float(result.horizontal_norm),
                        "endpoint_error": float(endpoint_error),
                        "ode_residual": float(jnp.maximum(residual.max_mu_abs, residual.max_sigma_abs)),
                        "gradient_norm": float(result.gradient_norm),
                        "hessian_min_eigenvalue": float(hessian_min),
                        "success": int(result.success_flag),
                        "solver_status": str(result.solver_result),
                    }
                )
            energies = jnp.asarray([item.energy for item in endpoint_results])
            omega_vectors = jnp.stack([item.omega_vec for item in endpoint_results])
            rows.append(
                {
                    "dimension": dimension,
                    "endpoint": endpoint_index,
                    "start": -1,
                    "condition": float(jnp.linalg.cond(sigma)),
                    "mu_norm": float(jnp.linalg.norm(mu)),
                    "energy": float(jnp.min(energies)),
                    "horizontal_residual": float(jnp.max(jnp.asarray([item.horizontal_norm for item in endpoint_results]))),
                    "endpoint_error": 0.0,
                    "ode_residual": 0.0,
                    "gradient_norm": float(jnp.max(jnp.asarray([item.gradient_norm for item in endpoint_results]))),
                    "hessian_min_eigenvalue": 0.0,
                    "success": int(jnp.all(jnp.asarray([item.success_flag for item in endpoint_results]))),
                    "solver_status": f"omega_spread={float(jnp.max(jnp.linalg.norm(omega_vectors - omega_vectors[0], axis=1))):.6g}",
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--endpoints", type=int, default=3)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("experiments/output/random_endpoint_scan.csv"))
    args = parser.parse_args()
    rows = scan(args.seed, tuple(args.dimensions), args.endpoints, args.starts)
    write_rows(args.output, rows)
    print(f"wrote {args.output}")
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()

