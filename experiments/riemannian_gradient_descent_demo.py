"""Demonstrate Fisher-Rao Riemannian gradient descent with the MVN expmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_fisher import _config as _config
from gaussian_fisher.riemannian_grad import rgd


def _print_result(name: str, result) -> None:
    monotone = bool(jnp.all(jnp.diff(result.values) <= 1e-10))
    accepted = int(result.step_sizes.shape[0])
    print(name)
    print(f"  initial objective: {float(result.values[0]):.12g}")
    print(f"  final objective:   {float(result.values[-1]):.12g}")
    print(f"  accepted steps:    {accepted}")
    print(f"  final grad norm:   {float(result.grad_norms[-1]):.12g}")
    print(f"  min eig Sigma:     {float(jnp.min(jnp.linalg.eigvalsh(result.sigma))):.12g}")
    print(f"  monotone:          {monotone}")
    print(f"  status:            {result.status}")


def mean_only_demo(step_size: float, n_steps: int):
    target = jnp.array([1.0, -0.5])
    mu0 = jnp.array([0.0, 0.0])
    sigma0 = jnp.array([[1.4, 0.2], [0.2, 0.9]])

    def objective(mu, sigma):
        del sigma
        diff = mu - target
        return 0.5 * diff @ diff

    return rgd(objective, mu0, sigma0, step_size=step_size, n_steps=n_steps)


def covariance_demo(step_size: float, n_steps: int):
    target = jnp.array([[1.2, 0.15], [0.15, 0.7]])
    target_inv = jnp.linalg.inv(target)
    mu0 = jnp.zeros(2)
    sigma0 = jnp.array([[2.0, 0.3], [0.3, 1.4]])

    def objective(mu, sigma):
        del mu
        sign, logdet = jnp.linalg.slogdet(sigma)
        return 0.5 * (jnp.trace(target_inv @ sigma) - logdet) + 0.0 * sign

    return rgd(objective, mu0, sigma0, step_size=step_size, n_steps=n_steps)


def nll_demo(seed: int, step_size: float, n_steps: int):
    key = jr.PRNGKey(seed)
    true_mu = jnp.array([0.8, -0.3])
    true_sigma = jnp.array([[0.7, 0.2], [0.2, 0.5]])
    raw = jr.normal(key, (64, 2))
    samples = true_mu + raw @ jnp.linalg.cholesky(true_sigma).T
    mu0 = jnp.zeros(2)
    sigma0 = jnp.eye(2)

    def objective(mu, sigma):
        centered = samples - mu
        solved = jnp.linalg.solve(sigma, centered.T).T
        sign, logdet = jnp.linalg.slogdet(sigma)
        return 0.5 * samples.shape[0] * logdet + 0.5 * jnp.sum(centered * solved) + 0.0 * sign

    return rgd(objective, mu0, sigma0, step_size=step_size, n_steps=n_steps, backtracking=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=20)
    args = parser.parse_args()

    _print_result("mean-only objective", mean_only_demo(args.step_size, args.n_steps))
    _print_result("covariance objective", covariance_demo(args.step_size, args.n_steps))
    _print_result("negative log likelihood", nll_demo(args.seed, args.step_size, args.n_steps))


if __name__ == "__main__":
    main()
