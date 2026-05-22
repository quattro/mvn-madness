"""Compare Nielsen baselines with horizontal-lift distances on random endpoints."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from nielsen_common import condition_and_mean_norm, hilbert_covariance_proxy, horizontal_lift_distance, random_spd, write_rows

from gaussian_fisher.nielsen import nielsen_distance


def build_endpoint(key: jr.PRNGKey, dimension: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k0, k1, k2, k3 = jr.split(key, 4)
    mu0 = jr.normal(k0, (dimension,)) * 0.2
    mu1 = jr.normal(k1, (dimension,)) * 0.2
    sigma0 = random_spd(k2, dimension)
    sigma1 = random_spd(k3, dimension)
    return mu0, sigma0, mu1, sigma1


def named_endpoint(name: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if name == "scalar":
        return jnp.array([0.0]), jnp.eye(1), jnp.array([0.8]), jnp.array([[2.25]])
    if name == "zero_mean":
        return jnp.zeros(2), jnp.eye(2), jnp.zeros(2), jnp.array([[3.0, 0.4], [0.4, 1.5]])
    if name == "d2_noncommuting_moderate":
        return jnp.zeros(2), jnp.eye(2), jnp.array([0.4, -0.3]), jnp.array([[2.0, 0.25], [0.25, 0.8]])
    if name == "d3_random_moderate":
        return jnp.zeros(3), jnp.eye(3), jr.normal(jr.PRNGKey(3), (3,)) * 0.2, random_spd(jr.PRNGKey(4), 3)
    if name == "projection_conditioning":
        from diagnose_adversarial_endpoint import reproduce_endpoint

        mu, sigma, _condition, _mean_norm, _starts_key = reproduce_endpoint(0, 3)
        return jnp.zeros(3), jnp.eye(3), mu, sigma
    raise ValueError(f"unknown named endpoint: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--endpoints", type=int, default=10)
    parser.add_argument(
        "--named-cases",
        nargs="*",
        default=[],
        choices=["scalar", "zero_mean", "d2_noncommuting_moderate", "d3_random_moderate", "projection_conditioning"],
    )
    parser.add_argument("--n-steps", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--curves", nargs="+", default=["source", "natural", "expectation"])
    parser.add_argument("--output", type=Path, default=Path("experiments/output/nielsen_vs_horizontal_lift.csv"))
    args = parser.parse_args()

    rows = []
    key = jr.PRNGKey(args.seed)
    endpoint_index = 0
    endpoints = [(name, *named_endpoint(name)) for name in args.named_cases]
    for dimension in args.dimensions:
        for _ in range(args.endpoints):
            key, endpoint_key = jr.split(key)
            endpoints.append((f"random_d{dimension}_{endpoint_index}", *build_endpoint(endpoint_key, dimension)))
    for endpoint_name, mu0, sigma0, mu1, sigma1 in endpoints:
            dimension = mu0.shape[0]
            condition, mean_norm = condition_and_mean_norm(mu0, sigma0, mu1, sigma1)
            horizontal = horizontal_lift_distance(mu0, sigma0, mu1, sigma1)
            proxy = hilbert_covariance_proxy(sigma0, sigma1)
            for n_steps in args.n_steps:
                for curve in args.curves:
                    start = time.perf_counter()
                    try:
                        result = nielsen_distance(mu0, sigma0, mu1, sigma1, n_steps, curve=curve)
                        distance = float(result.distance)
                        status = result.status
                    except Exception as exc:
                        distance = math.nan
                        status = f"error:{type(exc).__name__}:{exc}"
                    rows.append(
                        {
                            "dimension": dimension,
                            "endpoint": endpoint_index,
                            "endpoint_name": endpoint_name,
                            "condition": condition,
                            "mean_norm": mean_norm,
                            "curve": curve,
                            "n_steps": n_steps,
                            "distance": distance,
                            "horizontal_lift_distance": horizontal["distance"],
                            "relative_difference_from_horizontal_lift": abs(distance - horizontal["distance"]) / horizontal["distance"] if math.isfinite(distance) else math.nan,
                            "hilbert_covariance_proxy": proxy,
                            "runtime_seconds": time.perf_counter() - start,
                            "horizontal_solver_status": horizontal["solver_status"],
                            "horizontal_residual": horizontal["horizontal_residual"],
                            "horizontal_ode_residual": horizontal["ode_residual"],
                            "horizontal_ode_residual_coordinates": horizontal["ode_residual_coordinates"],
                            "horizontal_precision_ode_delta_abs": horizontal["precision_ode_delta_abs"],
                            "horizontal_precision_ode_theta_abs": horizontal["precision_ode_theta_abs"],
                            "horizontal_precision_ode_delta_rel": horizontal["precision_ode_delta_rel"],
                            "horizontal_precision_ode_theta_rel": horizontal["precision_ode_theta_rel"],
                            "horizontal_lifted_endpoint_error": horizontal["lifted_endpoint_error"],
                            "horizontal_precision_endpoint_error": horizontal["precision_endpoint_error"],
                            "horizontal_covariance_endpoint_error": horizontal["covariance_endpoint_error"],
                            "horizontal_projection_failure_class": horizontal["projection_failure_class"],
                            "status": status,
                        }
                    )
            endpoint_index += 1
    write_rows(args.output, rows)
    print(f"wrote {args.output}")
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()
