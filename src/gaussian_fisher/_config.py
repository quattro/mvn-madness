"""Runtime configuration for Gaussian Fisher numerical experiments."""

from __future__ import annotations

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
