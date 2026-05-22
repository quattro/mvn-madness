"""Nielsen approximation convergence tables."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import jax.numpy as jnp

from nielsen_common import exact_distance_if_available, horizontal_lift_distance, random_spd, write_rows

from gaussian_fisher.nielsen import nielsen_distance


def endpoint_for_dimension(dimension: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if dimension == 1:
        return jnp.array([0.0]), jnp.eye(1), jnp.array([0.8]), jnp.array([[2.25]])
    mu0 = jnp.zeros(dimension)
    sigma0 = jnp.eye(dimension)
    mu1 = jnp.linspace(0.1, 0.4, dimension)
    sigma1 = random_spd(jnp.asarray([dimension, 123], dtype=jnp.uint32), dimension)
    return mu0, sigma0, mu1, sigma1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--n-steps", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--curves", nargs="+", default=["source", "natural", "expectation"])
    parser.add_argument("--output", type=Path, default=Path("experiments/output/nielsen_convergence.csv"))
    args = parser.parse_args()

    mu0, sigma0, mu1, sigma1 = endpoint_for_dimension(args.dimension)
    exact = exact_distance_if_available(mu0, sigma0, mu1, sigma1)
    horizontal = horizontal_lift_distance(mu0, sigma0, mu1, sigma1)
    rows = []
    for curve in args.curves:
        for n_steps in args.n_steps:
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
                    "dimension": args.dimension,
                    "curve": curve,
                    "n_steps": n_steps,
                    "nielsen_distance": distance,
                    "exact_distance": exact,
                    "horizontal_lift_distance": horizontal["distance"],
                    "abs_diff_from_horizontal_lift": abs(distance - horizontal["distance"]) if math.isfinite(distance) else math.nan,
                    "runtime_seconds": time.perf_counter() - start,
                    "status": status,
                }
            )
    write_rows(args.output, rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

