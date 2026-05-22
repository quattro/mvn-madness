import jax.numpy as jnp

from experiments.d2_uniqueness_scan import scan


def test_d2_uniqueness_scan_cases_have_single_apparent_minimum_on_grid():
    cases = [
        ((0.5, 2.0), (0.5, 0.7)),
        ((0.5, 2.0), (3.0, 2.0)),
        ((0.01, 100.0), (1.0, 1.0)),
        ((0.05, 20.0), (4.0, -2.0)),
    ]
    omegas = jnp.linspace(-8.0, 8.0, 801)

    for scales, mean in cases:
        sigma = jnp.diag(jnp.asarray(scales))
        mu = jnp.asarray(mean)
        result = scan(mu, sigma, omegas)

        assert int(result["num_local_minima"]) >= 1
        assert bool(result["refined_hessian"] > 0.0)
        assert float(result["refined_horizontal_norm"]) < 1e-7
        assert float(result["optimizer_spread"]) < 1e-5
