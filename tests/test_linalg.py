import jax.numpy as jnp

from gaussian_fisher.linalg import expm_sym, invsqrtm_spd, logm_spd, matrix_power_spd, skew, sqrtm_spd


def test_spd_square_root_and_inverse_square_root_roundtrip():
    matrix = jnp.array([[3.0, 0.5], [0.5, 2.0]])

    root = sqrtm_spd(matrix)
    invroot = invsqrtm_spd(matrix)

    assert jnp.allclose(root @ root, matrix, atol=1e-6)
    assert jnp.allclose(invroot @ matrix @ invroot, jnp.eye(2), atol=1e-6)


def test_log_exp_roundtrip_for_spd_matrix():
    matrix = jnp.array([[2.5, 0.2], [0.2, 1.4]])

    assert jnp.allclose(expm_sym(logm_spd(matrix)), matrix, atol=1e-6)


def test_matrix_power_matches_identity_cases():
    matrix = jnp.array([[4.0, 0.0], [0.0, 9.0]])

    assert jnp.allclose(matrix_power_spd(matrix, 0.0), jnp.eye(2), atol=1e-6)
    assert jnp.allclose(matrix_power_spd(matrix, 1.0), matrix, atol=1e-6)


def test_skew_projection_is_skew_symmetric():
    matrix = jnp.array([[1.0, 3.0], [5.0, 7.0]])

    projected = skew(matrix)

    assert jnp.allclose(projected + projected.T, jnp.zeros((2, 2)))

