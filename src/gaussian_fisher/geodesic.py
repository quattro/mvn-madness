"""Projection of gauge-matrix paths back to Gaussian coordinates."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import lineax as lx

from gaussian_fisher import _config as _config
from gaussian_fisher.kobayashi import gauge_matrix
from gaussian_fisher.linalg import expm_sym, inv_spd, logm_spd, symmetrize
from gaussian_fisher.normalize import denormalize_path, normalize_endpoint
from gaussian_fisher.solvers import minimize_gauge


class GeodesicPath(NamedTuple):
    t: jnp.ndarray
    mu: jnp.ndarray
    sigma: jnp.ndarray
    omega: jnp.ndarray


class PrecisionGeodesicPath(NamedTuple):
    t: jnp.ndarray
    delta: jnp.ndarray
    theta: jnp.ndarray
    omega: jnp.ndarray


def project_to_precision_gaussian(gauge: jnp.ndarray, dimension: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract ``(delta, Theta)`` from the projected precision block.

    The upper-left ``(d+1) x (d+1)`` block has coordinates
    ``[[Theta, delta], [delta.T, scalar]]``. This function avoids inverting
    ``Theta`` and is useful for diagnostics on ill-conditioned endpoints.
    """

    theta = symmetrize(gauge[:dimension, :dimension])
    delta = gauge[:dimension, dimension : dimension + 1].reshape((dimension,))
    return delta, theta


def solve_precision_mean(theta: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Solve ``Theta mu = delta`` with Lineax."""

    operator = lx.MatrixLinearOperator(theta, tags=lx.positive_semidefinite_tag)
    return lx.linear_solve(operator, delta, solver=lx.AutoLinearSolver(well_posed=True)).value


def project_gauge_matrix(gauge: jnp.ndarray, dimension: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract ``(mu, sigma)`` from a matrix in the lifted Gaussian form."""

    delta, theta = project_to_precision_gaussian(gauge, dimension)
    mu = solve_precision_mean(theta, delta)
    sigma = inv_spd(theta)
    return mu, sigma


def project_to_gaussian(gauge: jnp.ndarray, dimension: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    return project_gauge_matrix(gauge, dimension)


def lifted_geodesic(mu: jnp.ndarray, sigma: jnp.ndarray, omega_star: jnp.ndarray, ts: jnp.ndarray) -> jnp.ndarray:
    log_g = logm_spd(gauge_matrix(mu, sigma, omega_star))

    def step(carry: None, time: jnp.ndarray) -> tuple[None, jnp.ndarray]:
        return carry, expm_sym(time * log_g)

    _, gauges = jax.lax.scan(step, None, ts)
    return gauges


def projected_geodesic_path(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    t: jnp.ndarray,
    *,
    omega: jnp.ndarray | None = None,
) -> GeodesicPath:
    """Return the projected path ``exp(t log G_omega)`` in normalized coordinates."""

    if omega is None:
        omega = minimize_gauge(mu, sigma).omega_star
    gauges = lifted_geodesic(mu, sigma, omega, t)
    mus, sigmas = jax.vmap(lambda gauge: project_gauge_matrix(gauge, mu.shape[0]))(gauges)
    return GeodesicPath(t, mus, sigmas, omega)


def projected_precision_geodesic_path(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    t: jnp.ndarray,
    *,
    omega: jnp.ndarray | None = None,
) -> PrecisionGeodesicPath:
    """Return the projected precision-coordinate path ``(delta(t), Theta(t))``.

    This avoids inverting ``Theta(t)`` and is the preferred diagnostic path for
    endpoints close to the SPD boundary.
    """

    if omega is None:
        omega = minimize_gauge(mu, sigma).omega_star
    gauges = lifted_geodesic(mu, sigma, omega, t)
    deltas, thetas = jax.vmap(lambda gauge: project_to_precision_gaussian(gauge, mu.shape[0]))(gauges)
    return PrecisionGeodesicPath(t, deltas, thetas, omega)


def gaussian_geodesic_normalized(mu: jnp.ndarray, sigma: jnp.ndarray, ts: jnp.ndarray) -> GeodesicPath:
    result = minimize_gauge(mu, sigma)
    return projected_geodesic_path(mu, sigma, ts, omega=result.omega_star)


def gaussian_geodesic_precision_normalized(mu: jnp.ndarray, sigma: jnp.ndarray, ts: jnp.ndarray) -> PrecisionGeodesicPath:
    result = minimize_gauge(mu, sigma)
    return projected_precision_geodesic_path(mu, sigma, ts, omega=result.omega_star)


def gaussian_geodesic(
    mu0: jnp.ndarray,
    sigma0: jnp.ndarray,
    mu1: jnp.ndarray,
    sigma1: jnp.ndarray,
    ts: jnp.ndarray,
) -> GeodesicPath:
    endpoint = normalize_endpoint(mu0, sigma0, mu1, sigma1)
    normalized = gaussian_geodesic_normalized(endpoint.mu, endpoint.sigma, ts)
    mus, sigmas = denormalize_path(endpoint, normalized.mu, normalized.sigma)
    return GeodesicPath(ts, mus, sigmas, normalized.omega)


def gaussian_geodesic_precision(
    mu0: jnp.ndarray,
    sigma0: jnp.ndarray,
    mu1: jnp.ndarray,
    sigma1: jnp.ndarray,
    ts: jnp.ndarray,
) -> PrecisionGeodesicPath:
    """Return a normalized precision-coordinate geodesic path.

    The returned ``(delta, theta)`` coordinates are for the affine-normalized
    endpoint problem ``(0, I) -> (mu, Sigma)``. Use this API for stable endpoint
    diagnostics when covariance inversion is ill-conditioned.
    """

    endpoint = normalize_endpoint(mu0, sigma0, mu1, sigma1)
    return gaussian_geodesic_precision_normalized(endpoint.mu, endpoint.sigma, ts)
