"""Compare the speculative P,S reduction with validated gauge minimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_fisher.ps_reduction import solve_ps_reduction
from gaussian_fisher.solvers import minimize_gauge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-unimplemented", action="store_true")
    args = parser.parse_args()
    mu = jnp.array([0.3, -0.2])
    sigma = jnp.array([[1.4, 0.1], [0.1, 1.2]])
    variational = minimize_gauge(mu, sigma)
    print(f"variational horizontal residual {float(variational.horizontal_norm):.8g}")
    try:
        solve_ps_reduction(mu, sigma)
    except NotImplementedError as exc:
        print(f"P,S reduction not validated: {exc}")
        if args.fail_on_unimplemented:
            raise


if __name__ == "__main__":
    main()

