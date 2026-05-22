"""Hilbert/SPD-cone proxy distances.

These distances are proxy distances and should not be reported as Fisher-Rao
distances. The generic SPD Hilbert projective distance is implemented without
assuming a Gaussian embedding convention.
"""

from __future__ import annotations

import jax.numpy as jnp

from gaussian_fisher import _config as _config
from gaussian_fisher.linalg import invsqrtm_spd, symmetrize


def hilbert_spd_distance(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Hilbert projective distance on SPD matrices."""

    whitened = invsqrtm_spd(a) @ symmetrize(b) @ invsqrtm_spd(a)
    eigenvalues = jnp.linalg.eigvalsh(symmetrize(whitened))
    return jnp.log(jnp.max(eigenvalues) / jnp.min(eigenvalues))


def gaussian_to_spd_embedding(_mu: jnp.ndarray, _sigma: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("Gaussian Hilbert-cone embedding convention is not specified")

