"""Small solver wrappers for horizontal gauge searches."""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx

from gaussian_fisher import _config as _config
from gaussian_fisher.energy import autodiff_gradient, energy, energy_from_skew_vector, horizontal_residual
from gaussian_fisher.linalg import skew, skew_from_vector, vector_from_skew


class GaugeSolverConfig(eqx.Module):
    """Static solver settings for horizontal gauge minimization."""

    learning_rate: float = eqx.field(static=True, default=0.25)
    max_steps: int = eqx.field(static=True, default=256)
    tolerance: float = eqx.field(static=True, default=1e-8)


class SolverResult(NamedTuple):
    omega_star: jnp.ndarray
    omega_vec: jnp.ndarray
    energy: jnp.ndarray
    gradient_norm: jnp.ndarray
    horizontal_norm: jnp.ndarray
    num_steps: int
    solver_result: object
    success_flag: bool

    @property
    def omega(self) -> jnp.ndarray:
        return self.omega_star

    @property
    def iterations(self) -> int:
        return self.num_steps

    @property
    def converged(self) -> bool:
        return self.success_flag


def _initial_vector(mu: jnp.ndarray, omega0: jnp.ndarray | None) -> jnp.ndarray:
    dimension = mu.shape[0]
    if omega0 is None:
        return jnp.zeros((dimension * (dimension - 1) // 2,), dtype=mu.dtype)
    if omega0.ndim == 1:
        return omega0
    return vector_from_skew(skew(omega0))


def _minimizer(name: str, config: GaugeSolverConfig) -> optx.AbstractMinimiser:
    if name == "bfgs":
        return optx.BFGS(rtol=config.tolerance, atol=config.tolerance)
    if name in {"gradient_descent", "gd"}:
        return optx.GradientDescent(config.learning_rate, rtol=config.tolerance, atol=config.tolerance)
    if name in {"nonlinear_cg", "cg"}:
        return optx.NonlinearCG(rtol=config.tolerance, atol=config.tolerance)
    raise ValueError(f"unknown minimizer: {name}")


def _root_solver(name: str, tolerance: float):
    if name == "newton":
        return optx.Newton(rtol=tolerance, atol=tolerance)
    if name == "chord":
        return optx.Chord(rtol=tolerance, atol=tolerance)
    if name in {"levenberg_marquardt", "lm"}:
        return optx.LevenbergMarquardt(rtol=tolerance, atol=tolerance)
    raise ValueError(f"unknown root solver: {name}")


def _result_from_solution(mu: jnp.ndarray, sigma: jnp.ndarray, solution: optx.Solution) -> SolverResult:
    omega_vec = solution.value
    omega = skew_from_vector(omega_vec, mu.shape[0])
    gradient_norm = jnp.linalg.norm(autodiff_gradient(mu, sigma, omega_vec))
    horizontal_norm = jnp.linalg.norm(horizontal_residual(mu, sigma, omega_vec))
    tolerance_success = bool(jnp.isfinite(gradient_norm) & jnp.isfinite(horizontal_norm))
    return SolverResult(
        omega,
        omega_vec,
        energy(mu, sigma, omega_vec),
        gradient_norm,
        horizontal_norm,
        int(solution.stats["num_steps"]),
        solution.result,
        tolerance_success,
    )


def minimize_gauge(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    omega0: jnp.ndarray | None = None,
    *,
    solver: str = "bfgs",
    max_steps: int = 256,
    tolerance: float = 1e-8,
    learning_rate: float = 0.25,
) -> SolverResult:
    """Minimize the Kobayashi fiber energy with Optimistix."""

    config = GaugeSolverConfig(learning_rate, max_steps, tolerance)
    params0 = _initial_vector(mu, omega0)
    solution = optx.minimise(
        lambda params, args: energy_from_skew_vector(params, *args),
        _minimizer(solver, config),
        params0,
        args=(mu, sigma),
        max_steps=config.max_steps,
        throw=False,
    )
    result = _result_from_solution(mu, sigma, solution)
    return result._replace(success_flag=bool(result.horizontal_norm <= tolerance or result.gradient_norm <= tolerance))


def solve_horizontal_root(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    omega0: jnp.ndarray | None = None,
    *,
    solver: str = "newton",
    max_steps: int = 256,
    tolerance: float = 1e-8,
) -> SolverResult:
    """Solve the horizontal residual equation with Optimistix."""

    params0 = _initial_vector(mu, omega0)
    solution = optx.root_find(
        lambda params, args: horizontal_residual(*args, params),
        _root_solver(solver, tolerance),
        params0,
        args=(mu, sigma),
        max_steps=max_steps,
        throw=False,
    )
    result = _result_from_solution(mu, sigma, solution)
    return result._replace(success_flag=bool(result.horizontal_norm <= tolerance))


class LegacySolverResult(NamedTuple):
    omega: jnp.ndarray
    energy: jnp.ndarray
    gradient_norm: jnp.ndarray
    iterations: int
    converged: bool


def solve_horizontal_gauge(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    *,
    initial_omega: jnp.ndarray | None = None,
    step_size: float = 0.25,
    max_steps: int = 256,
    tolerance: float = 1e-8,
) -> LegacySolverResult:
    """Projected gradient descent for the skew gauge variable.

    The state is represented in skew-coordinate form and minimized with
    Optimistix gradient descent. The returned matrix is projected back onto
    ``so(d)`` before reporting.
    """

    result = minimize_gauge(
        mu,
        sigma,
        initial_omega,
        solver="gradient_descent",
        max_steps=max_steps,
        tolerance=tolerance,
        learning_rate=step_size,
    )
    return LegacySolverResult(result.omega_star, result.energy, result.gradient_norm, result.num_steps, result.success_flag)
