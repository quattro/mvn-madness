import jax.numpy as jnp

from gaussian_fisher.hilbert_proxy import hilbert_spd_distance


def test_hilbert_spd_distance_invariants():
    a = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    b = jnp.array([[1.5, 0.1], [0.1, 3.0]])

    assert jnp.allclose(hilbert_spd_distance(a, a), 0.0, atol=1e-12)
    assert jnp.allclose(hilbert_spd_distance(a, b), hilbert_spd_distance(b, a), atol=1e-12)
    assert jnp.allclose(hilbert_spd_distance(2.0 * a, 7.0 * b), hilbert_spd_distance(a, b), atol=1e-12)
    assert hilbert_spd_distance(a, b) >= 0.0

