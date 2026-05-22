"""Compact Nielsen baseline smoke comparison."""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from nielsen_common import exact_distance_if_available, horizontal_lift_distance, random_spd

from gaussian_fisher.nielsen import nielsen_distance


def cases() -> list[tuple[str, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    return [
        ("scalar", jnp.array([0.0]), jnp.eye(1), jnp.array([0.8]), jnp.array([[2.25]])),
        ("zero_mean_d2", jnp.zeros(2), jnp.eye(2), jnp.zeros(2), jnp.array([[3.0, 0.4], [0.4, 1.5]])),
        ("noncommuting_d2", jnp.zeros(2), jnp.eye(2), jnp.array([0.4, -0.3]), jnp.array([[2.0, 0.25], [0.25, 0.8]])),
        ("random_d3", jnp.zeros(3), jnp.eye(3), jr.normal(jr.PRNGKey(3), (3,)) * 0.2, random_spd(jr.PRNGKey(4), 3)),
    ]


def main() -> None:
    curves = ("source", "natural", "expectation")
    print("case,curve,n_steps,nielsen_distance,horizontal_lift_distance,exact_distance,status")
    for name, mu0, sigma0, mu1, sigma1 in cases():
        horizontal = horizontal_lift_distance(mu0, sigma0, mu1, sigma1)
        exact = exact_distance_if_available(mu0, sigma0, mu1, sigma1)
        for curve in curves:
            try:
                result = nielsen_distance(mu0, sigma0, mu1, sigma1, 64, curve=curve)
                distance = float(result.distance)
                status = result.status
            except Exception as exc:
                distance = math.nan
                status = f"error:{type(exc).__name__}:{exc}"
            print(f"{name},{curve},64,{distance},{horizontal['distance']},{exact},{status}")


if __name__ == "__main__":
    main()

