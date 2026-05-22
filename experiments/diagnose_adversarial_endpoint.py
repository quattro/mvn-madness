"""Deep diagnostics for one adversarial random-multistart endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from adversarial_common import (
    endpoint_error_and_ode_residual,
    hessian_matrix,
    precision_projection_diagnostics,
    random_mean_with_norm,
    random_spd_with_condition,
    write_rows,
)

from gaussian_fisher.energy import hessian_vector_autodiff
from gaussian_fisher.geodesic import lifted_geodesic, projected_geodesic_path
from gaussian_fisher.kobayashi import gauge_matrix, log_gauge_matrix
from gaussian_fisher.solvers import minimize_gauge, solve_horizontal_root


def read_endpoint_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def select_endpoint(rows: list[dict[str, str]], endpoint: int | None) -> list[dict[str, str]]:
    if endpoint is None:
        worst = max(rows, key=lambda row: float(row.get("horizontal_residual", "nan")))
        endpoint = int(float(worst["endpoint"]))
    selected = [row for row in rows if int(float(row["endpoint"])) == endpoint]
    if not selected:
        raise ValueError(f"endpoint {endpoint} not found")
    return selected


def reproduce_endpoint(seed: int, dimension: int) -> tuple[jnp.ndarray, jnp.ndarray, float, float, jnp.ndarray]:
    key = jr.PRNGKey(seed)
    k_cond, k_sigma, k_mean_norm, k_mu, k_starts = jr.split(key, 5)
    condition = float(10.0 ** jr.uniform(k_cond, (), minval=0.0, maxval=6.0))
    mean_norm = float(10.0 ** jr.uniform(k_mean_norm, (), minval=-3.0, maxval=2.0))
    sigma = random_spd_with_condition(k_sigma, dimension, condition)
    mu = random_mean_with_norm(k_mu, dimension, mean_norm)
    return mu, sigma, condition, mean_norm, k_starts


def original_starts(start_key: jnp.ndarray, dimension: int, starts: int) -> jnp.ndarray:
    size = dimension * (dimension - 1) // 2
    return jnp.stack([jr.normal(key, (size,)) * 5.0 for key in jr.split(start_key, starts)])


def solve_one(label: str, start_index: int, mu: jnp.ndarray, sigma: jnp.ndarray, omega0: jnp.ndarray, *, solver: str, max_steps: int, tolerance: float, learning_rate: float) -> dict[str, Any]:
    try:
        result = minimize_gauge(
            mu,
            sigma,
            omega0,
            solver=solver,
            max_steps=max_steps,
            tolerance=tolerance,
            learning_rate=learning_rate,
        )
        endpoint_error, ode_residual = endpoint_error_and_ode_residual(mu, sigma, result.omega_star, samples=51)
        precision_diagnostics = precision_projection_diagnostics(mu, sigma, result.omega_star)
        hessian = hessian_matrix(mu, sigma, result.omega_vec)
        if hessian.shape == (0, 0):
            hessian_values = jnp.asarray([])
            hessian_vectors = jnp.zeros_like(hessian)
            hessian_min = math.inf
        else:
            hessian_values, hessian_vectors = jnp.linalg.eigh(0.5 * (hessian + hessian.T))
            hessian_min = float(hessian_values[0])
        return {
            "label": label,
            "start_index": start_index,
            "solver": solver,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "learning_rate": learning_rate,
            "omega0": omega0,
            "omega_vec": result.omega_vec,
            "omega_star": result.omega_star,
            "energy": float(result.energy),
            "horizontal_residual": float(result.horizontal_norm),
            "endpoint_error": endpoint_error,
            **precision_diagnostics,
            "ode_residual": ode_residual,
            "gradient_norm": float(result.gradient_norm),
            "hessian_matrix": hessian,
            "hessian_eigenvalues": hessian_values,
            "hessian_eigenvectors": hessian_vectors,
            "hessian_min_eigenvalue": hessian_min,
            "success": int(result.success_flag),
            "solver_status": str(result.solver_result),
        }
    except Exception as exc:
        return {
            "label": label,
            "start_index": start_index,
            "solver": solver,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "learning_rate": learning_rate,
            "omega0": omega0,
            "omega_vec": jnp.full_like(omega0, jnp.nan),
            "omega_star": jnp.full((mu.shape[0], mu.shape[0]), jnp.nan),
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
            "hessian_matrix": jnp.full((omega0.shape[0], omega0.shape[0]), jnp.nan),
            "hessian_eigenvalues": jnp.full((omega0.shape[0],), jnp.nan),
            "hessian_eigenvectors": jnp.full((omega0.shape[0], omega0.shape[0]), jnp.nan),
            "hessian_min_eigenvalue": math.nan,
            "success": 0,
            "solver_status": f"code_error:{type(exc).__name__}:{exc}",
        }


def solve_root_one(label: str, start_index: int, mu: jnp.ndarray, sigma: jnp.ndarray, omega0: jnp.ndarray, *, solver: str, max_steps: int, tolerance: float) -> dict[str, Any]:
    try:
        result = solve_horizontal_root(mu, sigma, omega0, solver=solver, max_steps=max_steps, tolerance=tolerance)
        endpoint_error, ode_residual = endpoint_error_and_ode_residual(mu, sigma, result.omega_star, samples=51)
        precision_diagnostics = precision_projection_diagnostics(mu, sigma, result.omega_star)
        hessian = hessian_matrix(mu, sigma, result.omega_vec)
        hessian_values, hessian_vectors = jnp.linalg.eigh(0.5 * (hessian + hessian.T))
        return {
            "label": label,
            "start_index": start_index,
            "solver": f"root_{solver}",
            "max_steps": max_steps,
            "tolerance": tolerance,
            "learning_rate": math.nan,
            "omega0": omega0,
            "omega_vec": result.omega_vec,
            "omega_star": result.omega_star,
            "energy": float(result.energy),
            "horizontal_residual": float(result.horizontal_norm),
            "endpoint_error": endpoint_error,
            **precision_diagnostics,
            "ode_residual": ode_residual,
            "gradient_norm": float(result.gradient_norm),
            "hessian_matrix": hessian,
            "hessian_eigenvalues": hessian_values,
            "hessian_eigenvectors": hessian_vectors,
            "hessian_min_eigenvalue": float(hessian_values[0]),
            "success": int(result.horizontal_norm < tolerance),
            "solver_status": str(result.solver_result),
        }
    except Exception as exc:
        return {
            "label": label,
            "start_index": start_index,
            "solver": f"root_{solver}",
            "max_steps": max_steps,
            "tolerance": tolerance,
            "learning_rate": math.nan,
            "omega0": omega0,
            "omega_vec": jnp.full_like(omega0, jnp.nan),
            "omega_star": jnp.full((mu.shape[0], mu.shape[0]), jnp.nan),
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
            "hessian_matrix": jnp.full((omega0.shape[0], omega0.shape[0]), jnp.nan),
            "hessian_eigenvalues": jnp.full((omega0.shape[0],), jnp.nan),
            "hessian_eigenvectors": jnp.full((omega0.shape[0], omega0.shape[0]), jnp.nan),
            "hessian_min_eigenvalue": math.nan,
            "success": 0,
            "solver_status": f"code_error:{type(exc).__name__}:{exc}",
        }


def row_from_solution(solution: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": solution["label"],
        "start_index": solution["start_index"],
        "solver": solution["solver"],
        "max_steps": solution["max_steps"],
        "tolerance": solution["tolerance"],
        "learning_rate": solution["learning_rate"],
        "energy": solution["energy"],
        "horizontal_residual": solution["horizontal_residual"],
        "endpoint_error": solution["endpoint_error"],
        "condition_sigma_target": solution["condition_sigma_target"],
        "condition_theta_projected": solution["condition_theta_projected"],
        "condition_gauge": solution["condition_gauge"],
        "lifted_endpoint_error": solution["lifted_endpoint_error"],
        "relative_theta_error": solution["relative_theta_error"],
        "relative_delta_error": solution["relative_delta_error"],
        "precision_endpoint_error": solution["precision_endpoint_error"],
        "relative_sigma_error": solution["relative_sigma_error"],
        "relative_mu_error": solution["relative_mu_error"],
        "covariance_endpoint_error": solution["covariance_endpoint_error"],
        "projection_failure_class": solution["projection_failure_class"],
        "ode_residual": solution["ode_residual"],
        "gradient_norm": solution["gradient_norm"],
        "hessian_min_eigenvalue": solution["hessian_min_eigenvalue"],
        "omega_norm": float(jnp.linalg.norm(solution["omega_vec"])),
        "success": solution["success"],
        "solver_status": solution["solver_status"],
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare_solutions(mu: jnp.ndarray, sigma: jnp.ndarray, solutions: list[dict[str, Any]], *, samples: int) -> list[dict[str, Any]]:
    rows = []
    finite = [solution for solution in solutions if jnp.all(jnp.isfinite(solution["omega_vec"]))]
    for i, left in enumerate(finite):
        for j, right in enumerate(finite):
            if j <= i:
                continue
            left_g = gauge_matrix(mu, sigma, left["omega_star"])
            right_g = gauge_matrix(mu, sigma, right["omega_star"])
            left_log = log_gauge_matrix(mu, sigma, left["omega_star"])
            right_log = log_gauge_matrix(mu, sigma, right["omega_star"])
            ts = jnp.linspace(0.0, 1.0, samples)
            left_path = projected_geodesic_path(mu, sigma, ts, omega=left["omega_star"])
            right_path = projected_geodesic_path(mu, sigma, ts, omega=right["omega_star"])
            omega_delta = right["omega_vec"] - left["omega_vec"]
            delta_norm = jnp.linalg.norm(omega_delta)
            smallest_vec = left["hessian_eigenvectors"][:, 0]
            alignment = jnp.abs(jnp.dot(omega_delta / delta_norm, smallest_vec)) if float(delta_norm) > 0.0 else jnp.nan
            rows.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "left_start": left["start_index"],
                    "right_start": right["start_index"],
                    "energy_abs_diff": abs(left["energy"] - right["energy"]),
                    "omega_distance": float(delta_norm),
                    "gauge_fro_distance": float(jnp.linalg.norm(left_g - right_g)),
                    "log_gauge_fro_distance": float(jnp.linalg.norm(left_log - right_log)),
                    "max_mu_path_distance": float(jnp.max(jnp.linalg.norm(left_path.mu - right_path.mu, axis=1))),
                    "max_sigma_path_distance": float(jnp.max(jnp.linalg.norm(left_path.sigma - right_path.sigma, axis=(1, 2)))),
                    "smallest_hessian_alignment": float(alignment),
                }
            )
    return rows


def defensive_rows(mu: jnp.ndarray, sigma: jnp.ndarray, solutions: list[dict[str, Any]], *, samples: int) -> list[dict[str, Any]]:
    rows = []
    ts = jnp.linspace(0.0, 1.0, samples)
    for solution in solutions:
        if not jnp.all(jnp.isfinite(solution["omega_vec"])):
            continue
        gauge = gauge_matrix(mu, sigma, solution["omega_star"])
        gauge_values = jnp.linalg.eigvalsh(gauge)
        lifted = lifted_geodesic(mu, sigma, solution["omega_star"], ts)
        lifted_values = jnp.stack([jnp.linalg.eigvalsh(item) for item in lifted])
        reconstructed = lifted[-1]
        rows.append(
            {
                "label": solution["label"],
                "start_index": solution["start_index"],
                "gauge_min_eigenvalue": float(jnp.min(gauge_values)),
                "gauge_max_eigenvalue": float(jnp.max(gauge_values)),
                "gauge_condition": float(jnp.max(gauge_values) / jnp.min(gauge_values)),
                "path_min_eigenvalue": float(jnp.min(lifted_values)),
                "path_max_eigenvalue": float(jnp.max(lifted_values)),
                "endpoint_gauge_reconstruction_error": float(jnp.linalg.norm(reconstructed - gauge)),
            }
        )
    return rows


def capped_spectrum_rows(mu: jnp.ndarray, sigma: jnp.ndarray, starts: jnp.ndarray, caps: list[float], *, max_steps: int, tolerance: float) -> list[dict[str, Any]]:
    values, vectors = jnp.linalg.eigh(sigma)
    ratios = values / jnp.min(values)
    mean_direction = mu / jnp.linalg.norm(mu)
    rows = []
    for cap in caps:
        capped_values = jnp.minimum(ratios, cap)
        capped_sigma = vectors @ jnp.diag(capped_values) @ vectors.T
        capped_mu = jnp.linalg.norm(mu) * mean_direction
        solutions = [
            solve_one("spectrum_cap", i, capped_mu, capped_sigma, starts[i], solver="bfgs", max_steps=max_steps, tolerance=tolerance, learning_rate=0.25)
            for i in range(starts.shape[0])
        ]
        finite = [solution for solution in solutions if jnp.all(jnp.isfinite(solution["omega_vec"]))]
        if finite:
            energies = [solution["energy"] for solution in finite]
            omegas = jnp.stack([solution["omega_vec"] for solution in finite])
            best = min(finite, key=lambda solution: solution["energy"])
            omega_spread = float(jnp.max(jnp.linalg.norm(omegas - omegas[0], axis=1)))
            energy_spread = max(energies) - min(energies)
        else:
            best = None
            omega_spread = math.nan
            energy_spread = math.nan
        rows.append(
            {
                "cap": cap,
                "condition": float(jnp.linalg.cond(capped_sigma)),
                "mean_norm": float(jnp.linalg.norm(capped_mu)),
                "best_energy": best["energy"] if best else math.nan,
                "best_horizontal_residual": best["horizontal_residual"] if best else math.nan,
                "best_endpoint_error": best["endpoint_error"] if best else math.nan,
                "best_lifted_endpoint_error": best["lifted_endpoint_error"] if best else math.nan,
                "best_precision_endpoint_error": best["precision_endpoint_error"] if best else math.nan,
                "best_covariance_endpoint_error": best["covariance_endpoint_error"] if best else math.nan,
                "best_projection_failure_class": best["projection_failure_class"] if best else "none",
                "condition_theta_projected": best["condition_theta_projected"] if best else math.nan,
                "condition_gauge": best["condition_gauge"] if best else math.nan,
                "best_gradient_norm": best["gradient_norm"] if best else math.nan,
                "best_hessian_min_eigenvalue": best["hessian_min_eigenvalue"] if best else math.nan,
                "energy_spread": energy_spread,
                "omega_spread": omega_spread,
                "solver_statuses": "|".join(solution["solver_status"] for solution in solutions),
            }
        )
    return rows


def save_npz(path: Path, mu: jnp.ndarray, sigma: jnp.ndarray, starts: jnp.ndarray, baseline: list[dict[str, Any]], strict: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sigma_values, sigma_vectors = jnp.linalg.eigh(sigma)
    all_solutions = baseline + strict
    np.savez(
        path,
        mu=np.asarray(mu),
        sigma=np.asarray(sigma),
        sigma_eigenvalues=np.asarray(sigma_values),
        sigma_eigenvectors=np.asarray(sigma_vectors),
        random_starts=np.asarray(starts),
        labels=np.asarray([solution["label"] for solution in all_solutions]),
        start_indices=np.asarray([solution["start_index"] for solution in all_solutions]),
        omega_vectors=np.asarray([solution["omega_vec"] for solution in all_solutions]),
        omega_matrices=np.asarray([solution["omega_star"] for solution in all_solutions]),
        energies=np.asarray([solution["energy"] for solution in all_solutions]),
        horizontal_residuals=np.asarray([solution["horizontal_residual"] for solution in all_solutions]),
        endpoint_errors=np.asarray([solution["endpoint_error"] for solution in all_solutions]),
        hessian_matrices=np.asarray([solution["hessian_matrix"] for solution in all_solutions]),
        hessian_eigenvalues=np.asarray([solution["hessian_eigenvalues"] for solution in all_solutions]),
        hessian_eigenvectors=np.asarray([solution["hessian_eigenvectors"] for solution in all_solutions]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("experiments/output/adversarial_random_smoke.csv"))
    parser.add_argument("--endpoint", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/output/adversarial_endpoint_diagnostic"))
    parser.add_argument("--strict-max-steps", type=int, default=4096)
    parser.add_argument("--strict-tolerance", type=float, default=1e-12)
    parser.add_argument("--baseline-max-steps", type=int, default=768)
    parser.add_argument("--caps", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6])
    args = parser.parse_args()

    selected = select_endpoint(read_endpoint_rows(args.csv), args.endpoint)
    seed = int(float(selected[0]["seed"]))
    dimension = int(float(selected[0]["dimension"]))
    starts_count = max(int(float(row["start"])) for row in selected) + 1
    mu, sigma, condition, mean_norm, starts_key = reproduce_endpoint(seed, dimension)
    starts = original_starts(starts_key, dimension, starts_count)

    baseline = [
        solve_one("baseline", i, mu, sigma, starts[i], solver="bfgs", max_steps=args.baseline_max_steps, tolerance=1e-9, learning_rate=0.25)
        for i in range(starts_count)
    ]
    finite_baseline = [solution for solution in baseline if jnp.all(jnp.isfinite(solution["omega_vec"]))]
    best_previous = min(finite_baseline, key=lambda solution: solution["energy"]) if finite_baseline else None
    strict = [
        solve_one("strict_from_start", i, mu, sigma, starts[i], solver="bfgs", max_steps=args.strict_max_steps, tolerance=args.strict_tolerance, learning_rate=0.25)
        for i in range(starts_count)
    ]
    if best_previous is not None:
        strict.append(
            solve_one(
                "strict_from_best_previous",
                -1,
                mu,
                sigma,
                best_previous["omega_vec"],
                solver="bfgs",
                max_steps=args.strict_max_steps,
                tolerance=args.strict_tolerance,
                learning_rate=0.25,
            )
        )
    root_refined = []
    for solution in baseline:
        if jnp.all(jnp.isfinite(solution["omega_vec"])):
            root_refined.append(
                solve_root_one(
                    "root_newton_from_baseline",
                    solution["start_index"],
                    mu,
                    sigma,
                    solution["omega_vec"],
                    solver="newton",
                    max_steps=args.strict_max_steps,
                    tolerance=args.strict_tolerance,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_solutions = baseline + strict + root_refined
    write_rows(args.output_dir / "solutions.csv", [row_from_solution(solution) for solution in all_solutions])
    write_rows(args.output_dir / "pairwise_comparisons.csv", compare_solutions(mu, sigma, strict + root_refined, samples=101))
    write_rows(args.output_dir / "defensive_gauge_diagnostics.csv", defensive_rows(mu, sigma, strict + root_refined, samples=101))
    write_rows(
        args.output_dir / "spectrum_caps.csv",
        capped_spectrum_rows(mu, sigma, starts, args.caps, max_steps=args.strict_max_steps, tolerance=args.strict_tolerance),
    )
    save_npz(args.output_dir / "endpoint_arrays.npz", mu, sigma, starts, baseline, strict + root_refined)

    finite_strict = [solution for solution in strict + root_refined if jnp.all(jnp.isfinite(solution["omega_vec"]))]
    if finite_strict:
        strict_omegas = jnp.stack([solution["omega_vec"] for solution in finite_strict])
        strict_energies = [solution["energy"] for solution in finite_strict]
        strict_spread = float(jnp.max(jnp.linalg.norm(strict_omegas - strict_omegas[0], axis=1)))
        strict_energy_spread = max(strict_energies) - min(strict_energies)
        best_strict = min(finite_strict, key=lambda solution: solution["energy"])
        best_horizontal = min(finite_strict, key=lambda solution: solution["horizontal_residual"])
        best_endpoint = min(finite_strict, key=lambda solution: solution["endpoint_error"])
    else:
        strict_spread = math.nan
        strict_energy_spread = math.nan
        best_strict = None
        best_horizontal = None
        best_endpoint = None
    report = {
        "source_csv": str(args.csv),
        "seed": seed,
        "dimension": dimension,
        "starts": starts_count,
        "condition": float(jnp.linalg.cond(sigma)),
        "target_condition": condition,
        "mu_norm": float(jnp.linalg.norm(mu)),
        "requested_mean_norm": mean_norm,
        "strict_horizontal_below_1e-8": bool(best_horizontal and best_horizontal["horizontal_residual"] < 1e-8),
        "strict_endpoint_error_below_1e-6": bool(best_endpoint and best_endpoint["endpoint_error"] < 1e-6),
        "strict_optimizer_spread": strict_spread,
        "strict_energy_spread": strict_energy_spread,
        "energies_indistinguishable_at_1e-8": bool(math.isfinite(strict_energy_spread) and strict_energy_spread < 1e-8),
        "smaller_trust_radius_or_line_search": "not applicable: current Optimistix BFGS wrapper does not expose an initial trust radius or line-search step; strict reruns use higher max_steps/tighter tolerance and best-solution initialization",
        "best_energy_candidate": row_from_solution(best_strict) if best_strict else None,
        "best_horizontal_candidate": row_from_solution(best_horizontal) if best_horizontal else None,
        "best_endpoint_candidate": row_from_solution(best_endpoint) if best_endpoint else None,
    }
    report = json_safe(report)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
