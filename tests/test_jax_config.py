from jax import config

from gaussian_fisher import _config as _config


def test_jax_config_uses_x64_and_highest_matmul_precision():
    assert config.values["jax_enable_x64"] is True
    assert config.values["jax_default_matmul_precision"] == "highest"

