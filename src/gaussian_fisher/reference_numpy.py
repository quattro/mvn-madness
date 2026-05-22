"""Small NumPy/SciPy reference helpers for debugging JAX numerics."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def logm_spd_numpy(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * np.log(values)) @ vectors.T


def logm_spd_np(matrix: np.ndarray) -> np.ndarray:
    return logm_spd_numpy(matrix)


def sqrtm_spd_np(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * np.sqrt(values)) @ vectors.T


def build_G_np(mu: np.ndarray, sigma: np.ndarray, omega: np.ndarray) -> np.ndarray:
    dimension = mu.shape[0]
    theta = np.linalg.inv(sigma)
    top = np.concatenate([np.eye(dimension), np.zeros((dimension, 1)), np.zeros((dimension, dimension))], axis=1)
    middle = np.concatenate([mu[None, :], np.ones((1, 1)), np.zeros((1, dimension))], axis=1)
    bottom = np.concatenate([omega - 0.5 * np.outer(mu, mu), -mu[:, None], np.eye(dimension)], axis=1)
    lift = np.concatenate([top, middle, bottom], axis=0)
    diagonal = scipy.linalg.block_diag(theta, np.ones((1, 1)), sigma)
    return lift @ diagonal @ lift.T


def energy_numpy(gauge_matrix: np.ndarray) -> float:
    log_g = scipy.linalg.logm(gauge_matrix)
    return float(0.5 * np.sum(np.real(log_g) ** 2))


def energy_np(mu: np.ndarray, sigma: np.ndarray, omega: np.ndarray) -> float:
    return energy_numpy(build_G_np(mu, sigma, omega))
