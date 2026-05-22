"""Finite-difference checks for candidate Gaussian paths."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import lineax as lx

from gaussian_fisher import _config as _config
from gaussian_fisher.ode import geodesic_acceleration


class ODEResidualSummary(NamedTuple):
    max_mu_abs: jnp.ndarray
    max_sigma_abs: jnp.ndarray
    max_mu_rel: jnp.ndarray
    max_sigma_rel: jnp.ndarray


class PrecisionODEResidualSummary(NamedTuple):
    max_delta_abs: jnp.ndarray
    max_theta_abs: jnp.ndarray
    max_delta_rel: jnp.ndarray
    max_theta_rel: jnp.ndarray


def _solve_spd(matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    operator = lx.MatrixLinearOperator(matrix, tags=lx.positive_semidefinite_tag)
    solver = lx.AutoLinearSolver(well_posed=True)
    solve_vector = lambda vector: lx.linear_solve(operator, vector, solver=solver).value
    if rhs.ndim == 1:
        return solve_vector(rhs)
    return jax.vmap(solve_vector, in_axes=1, out_axes=1)(rhs)


def geodesic_ode_residual(mus: jnp.ndarray, sigmas: jnp.ndarray, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return interior residuals of the Gaussian geodesic ODE for sampled paths."""

    dt = t[1] - t[0]
    mu_dot = (mus[2:] - mus[:-2]) / (2.0 * dt)
    sigma_dot = (sigmas[2:] - sigmas[:-2]) / (2.0 * dt)
    mu_ddot_fd = (mus[2:] - 2.0 * mus[1:-1] + mus[:-2]) / (dt * dt)
    sigma_ddot_fd = (sigmas[2:] - 2.0 * sigmas[1:-1] + sigmas[:-2]) / (dt * dt)

    mu_ddot, sigma_ddot = jax.vmap(geodesic_acceleration)(mu_dot, sigmas[1:-1], sigma_dot)
    return mu_ddot_fd - mu_ddot, sigma_ddot_fd - sigma_ddot


def finite_difference_ode_residual(mus: jnp.ndarray, sigmas: jnp.ndarray, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return geodesic_ode_residual(mus, sigmas, t)


def max_residual_norm(mus: jnp.ndarray, sigmas: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    mu_residual, sigma_residual = finite_difference_ode_residual(mus, sigmas, t)
    return jnp.maximum(jnp.max(jnp.linalg.norm(mu_residual, axis=1)), jnp.max(jnp.linalg.norm(sigma_residual, axis=(1, 2))))


def residual_summary(mus: jnp.ndarray, sigmas: jnp.ndarray, t: jnp.ndarray, *, eps: float = 1e-12) -> ODEResidualSummary:
    """Return absolute and relative residual diagnostics for sampled paths."""

    dt = t[1] - t[0]
    mu_dot = (mus[2:] - mus[:-2]) / (2.0 * dt)
    sigma_dot = (sigmas[2:] - sigmas[:-2]) / (2.0 * dt)
    mu_ddot_fd = (mus[2:] - 2.0 * mus[1:-1] + mus[:-2]) / (dt * dt)
    sigma_ddot_fd = (sigmas[2:] - 2.0 * sigmas[1:-1] + sigmas[:-2]) / (dt * dt)
    mu_residual, sigma_residual = geodesic_ode_residual(mus, sigmas, t)

    mu_ddot, sigma_ddot = jax.vmap(geodesic_acceleration)(mu_dot, sigmas[1:-1], sigma_dot)

    mu_abs = jnp.linalg.norm(mu_residual, axis=1)
    sigma_abs = jnp.linalg.norm(sigma_residual, axis=(1, 2))
    mu_scale = jnp.maximum(jnp.linalg.norm(mu_ddot_fd, axis=1) + jnp.linalg.norm(mu_ddot, axis=1), eps)
    sigma_scale = jnp.maximum(
        jnp.linalg.norm(sigma_ddot_fd, axis=(1, 2)) + jnp.linalg.norm(sigma_ddot, axis=(1, 2)),
        eps,
    )
    return ODEResidualSummary(
        jnp.max(mu_abs),
        jnp.max(sigma_abs),
        jnp.max(mu_abs / mu_scale),
        jnp.max(sigma_abs / sigma_scale),
    )


def precision_geodesic_ode_residual(deltas: jnp.ndarray, thetas: jnp.ndarray, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return precision-coordinate residuals for sampled ``(delta, Theta)`` paths.

    With ``Theta = Sigma^{-1}``, ``delta = Theta @ mu``, and
    ``q = delta_dot - Theta_dot @ solve(Theta, delta)``, the transformed ODE is

    ``q_dot = 0``

    and

    ``Theta_ddot - Theta_dot @ solve(Theta, Theta_dot) - q q.T = 0``.
    """

    if deltas.shape[0] < 5:
        raise ValueError("at least five samples are required for precision residuals")
    dt = t[1] - t[0]
    delta_dot = (deltas[2:] - deltas[:-2]) / (2.0 * dt)
    theta_dot = (thetas[2:] - thetas[:-2]) / (2.0 * dt)
    theta_ddot = (thetas[2:] - 2.0 * thetas[1:-1] + thetas[:-2]) / (dt * dt)

    mus = jax.vmap(_solve_spd)(thetas[1:-1], deltas[1:-1])
    q = delta_dot - jnp.einsum("tij,tj->ti", theta_dot, mus)
    q_dot = (q[2:] - q[:-2]) / (2.0 * dt)

    theta_mid = thetas[2:-2]
    theta_dot_mid = theta_dot[1:-1]
    theta_ddot_mid = theta_ddot[1:-1]
    q_mid = q[1:-1]
    theta_dot_solve = jax.vmap(_solve_spd)(theta_mid, theta_dot_mid)
    theta_residual = (
        theta_ddot_mid
        - jnp.einsum("tij,tjk->tik", theta_dot_mid, theta_dot_solve)
        - jnp.einsum("ti,tj->tij", q_mid, q_mid)
    )
    return q_dot, theta_residual


def precision_residual_summary(deltas: jnp.ndarray, thetas: jnp.ndarray, t: jnp.ndarray, *, eps: float = 1e-12) -> PrecisionODEResidualSummary:
    """Return absolute and relative precision-coordinate ODE residual diagnostics."""

    delta_residual, theta_residual = precision_geodesic_ode_residual(deltas, thetas, t)
    dt = t[1] - t[0]
    delta_dot = (deltas[2:] - deltas[:-2]) / (2.0 * dt)
    theta_dot = (thetas[2:] - thetas[:-2]) / (2.0 * dt)
    mus = jax.vmap(_solve_spd)(thetas[1:-1], deltas[1:-1])
    q = delta_dot - jnp.einsum("tij,tj->ti", theta_dot, mus)
    theta_ddot = (thetas[2:] - 2.0 * thetas[1:-1] + thetas[:-2]) / (dt * dt)
    theta_mid = thetas[2:-2]
    theta_dot_mid = theta_dot[1:-1]
    theta_ddot_mid = theta_ddot[1:-1]
    q_mid = q[1:-1]
    theta_dot_solve = jax.vmap(_solve_spd)(theta_mid, theta_dot_mid)
    theta_transport = jnp.einsum("tij,tjk->tik", theta_dot_mid, theta_dot_solve)
    q_outer = jnp.einsum("ti,tj->tij", q_mid, q_mid)
    delta_abs = jnp.linalg.norm(delta_residual, axis=1)
    theta_abs = jnp.linalg.norm(theta_residual, axis=(1, 2))
    q_scale = jnp.maximum(jnp.max(jnp.linalg.norm(q, axis=1)) / jnp.maximum(t[-1] - t[0], eps), eps)
    theta_scale = jnp.maximum(
        jnp.linalg.norm(theta_ddot_mid, axis=(1, 2))
        + jnp.linalg.norm(theta_transport, axis=(1, 2))
        + jnp.linalg.norm(q_outer, axis=(1, 2)),
        eps,
    )
    return PrecisionODEResidualSummary(
        jnp.max(delta_abs),
        jnp.max(theta_abs),
        jnp.max(delta_abs / q_scale),
        jnp.max(theta_abs / theta_scale),
    )


def max_precision_residual_norm(deltas: jnp.ndarray, thetas: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    delta_residual, theta_residual = precision_geodesic_ode_residual(deltas, thetas, t)
    return jnp.maximum(jnp.max(jnp.linalg.norm(delta_residual, axis=1)), jnp.max(jnp.linalg.norm(theta_residual, axis=(1, 2))))
