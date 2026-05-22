"""Numerical tools for Gaussian Fisher-Rao geodesics."""

from gaussian_fisher import _config as _config
from gaussian_fisher.energy import energy, explicit_gradient
from gaussian_fisher.expmap import expmap, expmap_normalized, expmap_path, normalize_tangent
from gaussian_fisher.kobayashi import gauge_matrix, horizontal_block
from gaussian_fisher.normalize import NormalizedEndpoint, normalize_endpoint, unnormalize_path
from gaussian_fisher.riemannian_grad import RGDResult, euclidean_to_riemannian_gradient, rgd, riemannian_gradient_step

__all__ = [
    "NormalizedEndpoint",
    "RGDResult",
    "energy",
    "euclidean_to_riemannian_gradient",
    "explicit_gradient",
    "expmap",
    "expmap_normalized",
    "expmap_path",
    "gauge_matrix",
    "horizontal_block",
    "normalize_tangent",
    "normalize_endpoint",
    "rgd",
    "riemannian_gradient_step",
    "unnormalize_path",
]
