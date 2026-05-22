"""Gaussian Fisher-Rao geodesic ODE residuals and optional Diffrax integration."""

from __future__ import annotations

import jax.numpy as jnp

from gaussian_fisher import _config as _config
from gaussian_fisher.linalg import inv_spd, symmetrize


def geodesic_acceleration(mu_dot: jnp.ndarray, sigma: jnp.ndarray, sigma_dot: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(mu_ddot, sigma_ddot)`` implied by the Gaussian geodesic ODE."""

    sigma_inv = inv_spd(sigma)
    mu_ddot = sigma_dot @ sigma_inv @ mu_dot
    sigma_ddot = sigma_dot @ sigma_inv @ sigma_dot - jnp.outer(mu_dot, mu_dot)
    return mu_ddot, symmetrize(sigma_ddot)


def geodesic_rhs(_t: jnp.ndarray, state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], _args: None = None):
    mu, sigma, mu_dot, sigma_dot = state
    mu_ddot, sigma_ddot = geodesic_acceleration(mu_dot, sigma, sigma_dot)
    return mu_dot, sigma_dot, mu_ddot, sigma_ddot


def ode_rhs(t: jnp.ndarray, state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], args: None = None):
    return geodesic_rhs(t, state, args)


def integrate_geodesic_ode(
    initial_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ts: jnp.ndarray,
    *,
    dt0: float = 1e-3,
    solver=None,
):
    """Integrate the geodesic ODE with Diffrax when the optional dependency is installed."""

    try:
        import diffrax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("diffrax is required for integrate_geodesic_ode") from exc
    term = diffrax.ODETerm(geodesic_rhs)
    ode_solver = diffrax.Dopri5() if solver is None else solver
    saveat = diffrax.SaveAt(ts=ts)
    return diffrax.diffeqsolve(term, ode_solver, float(ts[0]), float(ts[-1]), dt0, initial_state, saveat=saveat)


def integrate_with_diffrax(*args, **kwargs):
    return integrate_geodesic_ode(*args, **kwargs)
