import jax.numpy as jnp
import pytest

from gaussian_fisher.ps_reduction import scalar_fisher_distance, solve_ps_reduction


def test_scalar_distance_is_symmetric():
    a = scalar_fisher_distance(jnp.array(0.0), jnp.array(1.0), jnp.array(1.0), jnp.array(2.0))
    b = scalar_fisher_distance(jnp.array(1.0), jnp.array(2.0), jnp.array(0.0), jnp.array(1.0))

    assert jnp.allclose(a, b, atol=1e-7)


@pytest.mark.xfail(reason="P,S reduction is explicitly speculative until validated against gauge minimization")
def test_ps_reduction_is_not_yet_validated():
    solve_ps_reduction()
