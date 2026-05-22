"""Two-dimensional horizontal-gauge uniqueness scans."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_fisher.energy import energy, explicit_gradient, hessian_vector_autodiff, horizontal_residual
from gaussian_fisher.solvers import minimize_gauge


def scan(mu: jnp.ndarray, sigma: jnp.ndarray, omegas: jnp.ndarray) -> dict[str, jnp.ndarray]:
    energies = jax.vmap(lambda omega: energy(mu, sigma, jnp.array([omega])))(omegas)
    gradients = jax.vmap(lambda omega: explicit_gradient(mu, sigma, jnp.array([omega]))[0])(omegas)
    hessians = jax.vmap(lambda omega: hessian_vector_autodiff(mu, sigma, jnp.array([omega]), jnp.array([1.0]))[0])(omegas)
    horizontal_norms = jax.vmap(lambda omega: jnp.linalg.norm(horizontal_residual(mu, sigma, jnp.array([omega]))))(omegas)
    global_minimum = jnp.argmin(energies)
    sign_changes = (gradients[:-1] * gradients[1:]) <= 0
    local_minima = (energies[1:-1] < energies[:-2]) & (energies[1:-1] < energies[2:])
    starts = jnp.array([omegas[0], omegas[global_minimum], omegas[-1]])
    refined_results = [minimize_gauge(mu, sigma, jnp.array([start]), max_steps=512, tolerance=1e-10) for start in starts]
    refined_energies = jnp.asarray([result.energy for result in refined_results])
    refined_best = int(jnp.argmin(refined_energies))
    refined = refined_results[refined_best]
    refined_hessian = hessian_vector_autodiff(mu, sigma, refined.omega_vec, jnp.ones_like(refined.omega_vec))[0]
    optimizer_spread = jnp.max(jnp.abs(jnp.asarray([result.omega_vec[0] for result in refined_results]) - refined.omega_vec[0]))
    return {
        "omega": omegas,
        "energy": energies,
        "energy_prime": gradients,
        "hessian": hessians,
        "horizontal_norm": horizontal_norms,
        "minimizer": omegas[global_minimum],
        "global_energy": energies[global_minimum],
        "global_hessian": hessians[global_minimum],
        "global_horizontal_norm": horizontal_norms[global_minimum],
        "refined_minimizer": refined.omega_vec[0],
        "refined_energy": refined.energy,
        "refined_hessian": refined_hessian,
        "refined_horizontal_norm": refined.horizontal_norm,
        "refined_gradient_norm": refined.gradient_norm,
        "optimizer_spread": optimizer_spread,
        "num_critical_brackets": jnp.sum(sign_changes),
        "num_local_minima": jnp.sum(local_minima),
    }


def write_scan(path: Path, result: dict[str, jnp.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["omega", "energy", "energy_prime", "hessian", "horizontal_norm"])
        for row in zip(result["omega"], result["energy"], result["energy_prime"], result["hessian"], result["horizontal_norm"], strict=True):
            writer.writerow([float(value) for value in row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=float, default=2.0)
    parser.add_argument("--s2", type=float, default=0.7)
    parser.add_argument("--u", type=float, default=0.4)
    parser.add_argument("--v", type=float, default=-0.3)
    parser.add_argument("--lo", type=float, default=-3.0)
    parser.add_argument("--hi", type=float, default=3.0)
    parser.add_argument("--n", type=int, default=301)
    parser.add_argument("--output", type=Path, default=Path("experiments/output/d2_uniqueness_scan.csv"))
    args = parser.parse_args()

    mu = jnp.array([args.u, args.v])
    sigma = jnp.diag(jnp.array([args.s1, args.s2]))
    omegas = jnp.linspace(args.lo, args.hi, args.n)
    result = scan(mu, sigma, omegas)
    write_scan(args.output, result)
    print(f"wrote {args.output}")
    print(f"minimizer approximately {float(result['minimizer']):.8g}")
    print(f"critical brackets {int(result['num_critical_brackets'])}")
    print(f"local minima {int(result['num_local_minima'])}")
    print(f"global hessian {float(result['global_hessian']):.8g}")
    print(f"global horizontal residual {float(result['global_horizontal_norm']):.8g}")
    print(f"refined minimizer {float(result['refined_minimizer']):.8g}")
    print(f"refined horizontal residual {float(result['refined_horizontal_norm']):.8g}")
    print(f"optimizer spread {float(result['optimizer_spread']):.8g}")


if __name__ == "__main__":
    main()
