"""Nielsen-style discretized Fisher-Rao approximation baselines.

These routines approximate Fisher-Rao distance by sampling a chosen curve
between two multivariate normal distributions and summing local segment lengths
approximated by ``sqrt(Jeffreys divergence)``. They are baselines, not exact
geodesic solvers.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from gaussian_fisher import _config as _config
from gaussian_fisher.linalg import inv_spd, symmetrize


class NielsenApproxResult(NamedTuple):
    distance: jnp.ndarray
    curve: str
    n_steps: int
    segment_lengths: jnp.ndarray
    status: str


def _kl_mvn_value(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray) -> jnp.ndarray:
    dimension = mu0.shape[0]
    solved_sigma = jnp.linalg.solve(sigma1, sigma0)
    diff = mu1 - mu0
    solved_diff = jnp.linalg.solve(sigma1, diff)
    _, logdet0 = jnp.linalg.slogdet(sigma0)
    _, logdet1 = jnp.linalg.slogdet(sigma1)
    return 0.5 * (jnp.trace(solved_sigma) + diff @ solved_diff - dimension + logdet1 - logdet0)


def kl_mvn(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray) -> jnp.ndarray:
    """KL divergence ``KL(N(mu0, sigma0) || N(mu1, sigma1))``."""

    mu0 = jnp.asarray(mu0)
    mu1 = jnp.asarray(mu1)
    sigma0 = symmetrize(jnp.asarray(sigma0))
    sigma1 = symmetrize(jnp.asarray(sigma1))
    if mu0.shape != mu1.shape:
        raise ValueError("mu0 and mu1 must have matching shapes")
    if sigma0.shape != sigma1.shape or sigma0.shape != (mu0.shape[0], mu0.shape[0]):
        raise ValueError("covariance shapes must match the mean dimension")

    sign0, _ = jnp.linalg.slogdet(sigma0)
    sign1, _ = jnp.linalg.slogdet(sigma1)
    if bool(sign0 <= 0) or bool(sign1 <= 0):
        raise ValueError("covariances must be positive definite")
    return _kl_mvn_value(mu0, sigma0, mu1, sigma1)


def jeffreys_mvn(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray) -> jnp.ndarray:
    """Jeffreys divergence between two multivariate normal distributions."""

    return kl_mvn(mu0, sigma0, mu1, sigma1) + kl_mvn(mu1, sigma1, mu0, sigma0)


def _jeffreys_mvn_value(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray) -> jnp.ndarray:
    return _kl_mvn_value(mu0, sigma0, mu1, sigma1) + _kl_mvn_value(mu1, sigma1, mu0, sigma0)


def source_to_natural(mu: jnp.ndarray, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    k = inv_spd(sigma)
    return k @ mu, k


def natural_to_source(h: jnp.ndarray, k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    sigma = inv_spd(k)
    return sigma @ h, sigma


def source_to_expectation(mu: jnp.ndarray, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return mu, symmetrize(sigma + jnp.outer(mu, mu))


def expectation_to_source(m: jnp.ndarray, big_m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return m, symmetrize(big_m - jnp.outer(m, m))


def _symmetrize_batch(matrices: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrices + jnp.swapaxes(matrices, -1, -2))


def _check_path_spd(sigmas: jnp.ndarray, curve: str) -> None:
    eigenvalues = jnp.linalg.eigvalsh(sigmas)
    bad_index = jnp.argmax(jnp.any(eigenvalues <= 0.0, axis=1))
    if bool(jnp.any(eigenvalues <= 0.0)):
        raise ValueError(f"{curve} curve produced a non-SPD covariance at index {int(bad_index)}")


def source_curve(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray, ts: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    mean_weights = ts[:, None]
    matrix_weights = ts[:, None, None]
    mus = (1.0 - mean_weights) * mu0 + mean_weights * mu1
    sigmas = _symmetrize_batch((1.0 - matrix_weights) * sigma0 + matrix_weights * sigma1)
    _check_path_spd(sigmas, "source")
    return mus, sigmas


def natural_curve(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray, ts: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    h0, k0 = source_to_natural(mu0, sigma0)
    h1, k1 = source_to_natural(mu1, sigma1)
    mean_weights = ts[:, None]
    matrix_weights = ts[:, None, None]
    hs = (1.0 - mean_weights) * h0 + mean_weights * h1
    ks = _symmetrize_batch((1.0 - matrix_weights) * k0 + matrix_weights * k1)
    sigmas = jax.vmap(inv_spd)(ks)
    mus = jnp.einsum("tij,tj->ti", sigmas, hs)
    _check_path_spd(sigmas, "natural")
    return mus, sigmas


def expectation_curve(mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray, ts: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    m0, big_m0 = source_to_expectation(mu0, sigma0)
    m1, big_m1 = source_to_expectation(mu1, sigma1)
    mean_weights = ts[:, None]
    matrix_weights = ts[:, None, None]
    mus = (1.0 - mean_weights) * m0 + mean_weights * m1
    big_ms = _symmetrize_batch((1.0 - matrix_weights) * big_m0 + matrix_weights * big_m1)
    sigmas = _symmetrize_batch(big_ms - jnp.einsum("ti,tj->tij", mus, mus))
    _check_path_spd(sigmas, "expectation")
    return mus, sigmas


def calvo_oller_curve(_mu0: jnp.ndarray, _sigma0: jnp.ndarray, _mu1: jnp.ndarray, _sigma1: jnp.ndarray, _ts: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    raise NotImplementedError("Calvo-Oller curve requires a documented embedding convention before implementation")


def discretized_jeffreys_length(mu_path: jnp.ndarray, sigma_path: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sum ``sqrt(Jeffreys divergence)`` along adjacent path points."""

    if mu_path.shape[0] != sigma_path.shape[0]:
        raise ValueError("mu_path and sigma_path must have the same number of samples")
    divergences = jax.vmap(_jeffreys_mvn_value)(mu_path[:-1], sigma_path[:-1], mu_path[1:], sigma_path[1:])
    segments = jnp.sqrt(jnp.maximum(divergences, 0.0))
    return jnp.sum(segments), segments


def _curve_points(curve: str, mu0: jnp.ndarray, sigma0: jnp.ndarray, mu1: jnp.ndarray, sigma1: jnp.ndarray, ts: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if curve == "source":
        return source_curve(mu0, sigma0, mu1, sigma1, ts)
    if curve == "natural":
        return natural_curve(mu0, sigma0, mu1, sigma1, ts)
    if curve == "expectation":
        return expectation_curve(mu0, sigma0, mu1, sigma1, ts)
    if curve == "calvo_oller":
        return calvo_oller_curve(mu0, sigma0, mu1, sigma1, ts)
    raise ValueError(f"unknown Nielsen curve: {curve}")


def nielsen_distance(
    mu0: jnp.ndarray,
    sigma0: jnp.ndarray,
    mu1: jnp.ndarray,
    sigma1: jnp.ndarray,
    n_steps: int,
    curve: str = "source",
) -> NielsenApproxResult:
    """Approximate Fisher-Rao distance by a discretized curve and Jeffreys segments."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)
    mus, sigmas = _curve_points(curve, mu0, sigma0, mu1, sigma1, ts)
    distance, segments = discretized_jeffreys_length(mus, sigmas)
    return NielsenApproxResult(distance, curve, n_steps, segments, "ok")


def nielsen_convergence_table(
    mu0: jnp.ndarray,
    sigma0: jnp.ndarray,
    mu1: jnp.ndarray,
    sigma1: jnp.ndarray,
    n_steps_list: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256),
    curves: tuple[str, ...] = ("source", "natural", "expectation"),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for curve in curves:
        for n_steps in n_steps_list:
            try:
                result = nielsen_distance(mu0, sigma0, mu1, sigma1, n_steps, curve=curve)
                rows.append(
                    {
                        "curve": curve,
                        "n_steps": n_steps,
                        "distance": float(result.distance),
                        "status": result.status,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "curve": curve,
                        "n_steps": n_steps,
                        "distance": float("nan"),
                        "status": f"error:{type(exc).__name__}:{exc}",
                    }
                )
    return rows
