"""Adversarial d=2 grid scans for nonuniqueness and branch failures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from adversarial_common import bool_int, unique_count, write_rows

from gaussian_fisher.energy import energy, explicit_gradient, hessian_vector_autodiff, horizontal_residual
from gaussian_fisher.solvers import minimize_gauge, solve_horizontal_root


RATIOS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
MEAN_NORMS = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
ANGLES = [0.0, math.pi / 12.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0, 5.0 * math.pi / 12.0, math.pi / 2.0]


def _grid_values(mu: jnp.ndarray, sigma: jnp.ndarray, omegas: jnp.ndarray) -> dict[str, jnp.ndarray]:
    return {
        "energy": jax.vmap(lambda omega: energy(mu, sigma, jnp.array([omega])))(omegas),
        "gradient": jax.vmap(lambda omega: explicit_gradient(mu, sigma, jnp.array([omega]))[0])(omegas),
        "hessian": jax.vmap(lambda omega: hessian_vector_autodiff(mu, sigma, jnp.array([omega]), jnp.array([1.0]))[0])(omegas),
        "horizontal": jax.vmap(lambda omega: jnp.linalg.norm(horizontal_residual(mu, sigma, jnp.array([omega]))))(omegas),
    }


def _near_boundary(indices: jnp.ndarray, count: int, margin_fraction: float) -> bool:
    if indices.size == 0:
        return False
    margin = max(2, int(count * margin_fraction))
    return bool(jnp.any(indices < margin) or jnp.any(indices >= count - margin))


def adaptive_scan(mu: jnp.ndarray, sigma: jnp.ndarray, *, points: int, initial_range: float, max_range: float) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], float, int]:
    current = initial_range
    expansions = 0
    while True:
        omegas = jnp.linspace(-current, current, points)
        values = _grid_values(mu, sigma, omegas)
        gradients = values["gradient"]
        minimum_index = int(jnp.argmin(values["energy"]))
        critical_indices = jnp.where((gradients[:-1] * gradients[1:]) <= 0.0)[0]
        if current >= max_range:
            return omegas, values, current, expansions
        if not _near_boundary(jnp.asarray([minimum_index]), points, 0.05) and not _near_boundary(critical_indices, points - 1, 0.05):
            return omegas, values, current, expansions
        current *= 2.0
        expansions += 1


def refine_roots(mu: jnp.ndarray, sigma: jnp.ndarray, omegas: jnp.ndarray, gradients: jnp.ndarray) -> list[Any]:
    bracket_indices = jnp.where((gradients[:-1] * gradients[1:]) <= 0.0)[0]
    roots = []
    for index in list(map(int, bracket_indices)):
        start = jnp.array([0.5 * (omegas[index] + omegas[index + 1])])
        try:
            roots.append(solve_horizontal_root(mu, sigma, start, max_steps=128, tolerance=1e-10))
        except Exception:
            pass
    return roots


def scan_endpoint(ratio: float, mean_norm: float, angle: float, *, points: int, initial_range: float, max_range: float) -> dict[str, Any]:
    sigma = jnp.diag(jnp.array([ratio, 1.0]))
    mu = mean_norm * jnp.array([math.cos(angle), math.sin(angle)])
    genuinely_noncommuting = not (abs(math.sin(angle)) < 1e-14 or abs(math.cos(angle)) < 1e-14 or abs(ratio - 1.0) < 1e-14)
    omegas, values, omega_range, expansions = adaptive_scan(mu, sigma, points=points, initial_range=initial_range, max_range=max_range)
    energies = values["energy"]
    gradients = values["gradient"]
    hessians = values["hessian"]
    horizontal = values["horizontal"]
    critical_brackets = jnp.where((gradients[:-1] * gradients[1:]) <= 0.0)[0]
    local_minima = jnp.where((energies[1:-1] < energies[:-2]) & (energies[1:-1] < energies[2:]))[0] + 1
    global_index = int(jnp.argmin(energies))
    starts = [float(omegas[global_index]), -omega_range, 0.0, omega_range]
    roots = refine_roots(mu, sigma, omegas, gradients)
    minimizers = []
    statuses = []
    for start in starts:
        try:
            result = minimize_gauge(mu, sigma, jnp.array([start]), max_steps=512, tolerance=1e-10)
            minimizers.append(result)
            statuses.append(str(result.solver_result))
        except Exception as exc:
            statuses.append(f"code_error:{type(exc).__name__}:{exc}")
    if minimizers:
        best = min(minimizers, key=lambda result: float(result.energy))
        optimizer_spread = max(abs(float(result.omega_vec[0] - best.omega_vec[0])) for result in minimizers)
        refined_omega = float(best.omega_vec[0])
        refined_energy = float(best.energy)
        refined_horizontal = float(best.horizontal_norm)
        refined_gradient = float(best.gradient_norm)
        refined_hessian = float(hessian_vector_autodiff(mu, sigma, best.omega_vec, jnp.ones_like(best.omega_vec))[0])
    else:
        refined_omega = math.nan
        refined_energy = math.nan
        refined_horizontal = math.nan
        refined_gradient = math.nan
        refined_hessian = math.nan
        optimizer_spread = math.nan
    root_omegas = [float(root.omega_vec[0]) for root in roots if root.success_flag]
    root_hessians = [float(hessian_vector_autodiff(mu, sigma, root.omega_vec, jnp.ones_like(root.omega_vec))[0]) for root in roots if root.success_flag]
    root_horizontal = [float(root.horizontal_norm) for root in roots if root.success_flag]
    unique_roots = unique_count(root_omegas, tolerance=1e-6)
    unique_minima = unique_count([float(result.omega_vec[0]) for result in minimizers], tolerance=1e-5) if minimizers else 0
    return {
        "scan_type": "d2_grid",
        "dimension": 2,
        "ratio": ratio,
        "s1": ratio,
        "s2": 1.0,
        "mean_norm": mean_norm,
        "angle": angle,
        "condition": max(ratio, 1.0 / ratio),
        "genuinely_noncommuting": bool_int(genuinely_noncommuting),
        "omega_range": omega_range,
        "range_expansions": expansions,
        "grid_points": points,
        "num_critical_brackets": int(critical_brackets.size),
        "num_local_minima": int(local_minima.size),
        "multiple_critical_points": bool_int(int(critical_brackets.size) > 1),
        "multiple_apparent_minima": bool_int(int(local_minima.size) > 1 or unique_minima > 1),
        "global_minimizer_grid": float(omegas[global_index]),
        "global_energy_grid": float(energies[global_index]),
        "global_hessian_grid": float(hessians[global_index]),
        "global_horizontal_grid": float(horizontal[global_index]),
        "refined_minimizer": refined_omega,
        "refined_energy": refined_energy,
        "refined_hessian": refined_hessian,
        "smallest_critical_hessian": min(root_hessians) if root_hessians else math.nan,
        "refined_horizontal_residual": refined_horizontal,
        "refined_gradient_norm": refined_gradient,
        "optimizer_spread": optimizer_spread,
        "root_count_refined": unique_roots,
        "max_root_horizontal_residual": max(root_horizontal) if root_horizontal else math.nan,
        "solver_status": "|".join(statuses),
    }


def build_rows(*, points: int, initial_range: float, max_range: float, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for ratio in RATIOS:
        for mean_norm in MEAN_NORMS:
            for angle in ANGLES:
                rows.append(scan_endpoint(ratio, mean_norm, angle, points=points, initial_range=initial_range, max_range=max_range))
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1001)
    parser.add_argument("--initial-range", type=float, default=50.0)
    parser.add_argument("--max-range", type=float, default=6400.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("experiments/output/adversarial_d2_grid.csv"))
    args = parser.parse_args()
    rows = build_rows(points=args.points, initial_range=args.initial_range, max_range=args.max_range, limit=args.limit)
    write_rows(args.output, rows)
    noncommuting = [row for row in rows if row["genuinely_noncommuting"]]
    multiple = sum(int(row["multiple_apparent_minima"]) for row in noncommuting)
    print(f"wrote {args.output}")
    print(f"endpoints {len(rows)}")
    print(f"genuinely noncommuting endpoints {len(noncommuting)}")
    print(f"noncommuting endpoints with multiple apparent minima {multiple}")


if __name__ == "__main__":
    main()

