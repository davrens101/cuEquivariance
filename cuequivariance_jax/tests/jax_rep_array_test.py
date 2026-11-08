# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

import cuequivariance as cue
import cuequivariance_jax as cuex


@cue.assume("SO3", cue.ir_mul)
def test_segments():
    x = cuex.RepArray("2x0 + 1", jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    x0, x1 = x.segments
    assert x0.shape == (1, 2)
    assert x1.shape == (3, 1)
    y = cuex.from_segments("2x0 + 1", [x0, x1], x.shape)
    assert x.irreps == y.irreps
    assert x.layout == y.layout
    assert jnp.allclose(x.array, y.array)


@cue.assume("SO3", cue.ir_mul)
def test_slice_by_mul():
    x = cuex.RepArray("2x0 + 1", jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    x = x.slice_by_mul[1:]
    assert x.irreps == cue.Irreps("0 + 1")
    assert x.layout == cue.ir_mul
    assert jnp.allclose(x.array, jnp.array([1.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        pytest.param(
            jnp.float64,
            marks=pytest.mark.skipif(
                not jax.config.x64_enabled, reason="x64 not enabled"
            ),
        ),
    ],
)
def test_transform_gradient_at_identity(dtype):
    x = cuex.RepArray({0: cue.SO3(1)}, jnp.array([1.0, 2.0, 3.0], dtype=dtype))
    v0 = jnp.zeros(3, dtype=dtype)

    jac_fwd = jax.jacfwd(lambda v: x.transform(v).array)(v0)
    jac_rev = jax.jacrev(lambda v: x.transform(v).array)(v0)

    expected = jnp.array(
        [[0.0, 3.0, -2.0], [-3.0, 0.0, 1.0], [2.0, -1.0, 0.0]], dtype=dtype
    )
    assert jac_fwd.dtype == dtype
    assert jac_rev.dtype == dtype
    assert not jnp.any(jnp.isnan(jac_fwd))
    assert not jnp.any(jnp.isnan(jac_rev))
    np.testing.assert_allclose(jac_fwd, expected, atol=1e-5)
    np.testing.assert_allclose(jac_rev, expected, atol=1e-5)

    hessian = jax.jacfwd(jax.jacrev(lambda v: x.transform(v).array))(v0)
    generators = cue.SO3(1).X
    products = np.einsum("aij,bjk,k->iab", generators, generators, np.asarray(x.array))
    expected_hessian = (products + products.transpose(0, 2, 1)) / 2
    np.testing.assert_allclose(hessian, expected_hessian, atol=1e-5)


def test_transform_integer_angle_matches_scipy_and_exp_map():
    rep = cue.SO3(2)
    x = cuex.RepArray({0: rep}, jnp.ones(rep.dim, dtype=jnp.float32))
    v_int = jnp.array([1, 0, 0])

    out = x.transform(v_int).array

    R_ref = scipy.linalg.expm(np.einsum("a,aij->ij", np.array([1.0, 0.0, 0.0]), rep.X))
    expected = R_ref @ np.ones(rep.dim, dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-5)

    R_exp_map = rep.exp_map(np.array([1.0, 0.0, 0.0]), np.zeros(0, dtype=int))
    np.testing.assert_allclose(
        out, R_exp_map @ np.ones(rep.dim, dtype=np.float32), atol=1e-5
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_transform_half_precision_angles(dtype):
    rep = cue.SO3(1)
    x = cuex.RepArray({0: rep}, jnp.array([1.0, 2.0, 3.0], dtype=dtype))
    v = jnp.array([0.1, 0.2, 0.3], dtype=dtype)
    out = x.transform(v).array
    assert not jnp.any(jnp.isnan(out))
    # Matches float32 reference evaluation
    x_f32 = cuex.RepArray({0: rep}, jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
    v_f32 = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    ref = x_f32.transform(v_f32).array
    np.testing.assert_allclose(out, ref, atol=2e-2)


def test_transform_large_finite_angle_orthogonal():
    rep = cue.SO3(1)
    x = cuex.RepArray({0: rep}, jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
    v_large = jnp.array([1e6, 0.0, 0.0], dtype=jnp.float32)
    out = x.transform(v_large).array
    assert not jnp.any(jnp.isnan(out))
    # Preserves L2 norm and exact x-coordinate invariance for rotation around x
    np.testing.assert_allclose(out[0], x.array[0], atol=1e-5)
    np.testing.assert_allclose(
        jnp.linalg.norm(out), jnp.linalg.norm(x.array), atol=1e-5
    )

    tangent = jax.jvp(
        lambda t: x.transform(v_large + t * jnp.array([1.0, 0.0, 0.0])).array,
        (jnp.array(0.0),),
        (jnp.array(1.0),),
    )[1]
    expected_tangent = rep.X[0] @ np.asarray(out)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-5, atol=1e-5)


def test_transform_complex_angle_projection():
    x = cuex.RepArray({0: cue.SU2(0.5)}, jnp.array([1.0, 0.0]))

    def transform(t):
        return x.transform(jnp.array([0.0, 1.0 + 1j * t, 0.0])).array

    # eigh projects iA onto its Hermitian part; this imaginary perturbation
    # contributes nothing to that projection.
    np.testing.assert_allclose(jax.jacfwd(transform)(jnp.array(0.0)), 0, atol=1e-6)


def test_transform_batched_rotation_gradients():
    rep = cue.SO3(1)
    x = cuex.RepArray({0: rep}, jnp.array([1.0, 2.0, 3.0]))
    weights = jnp.array([0.5, -0.25, 2.0])
    angles = jnp.array([0.0, 0.125, 10.0])

    def loss(t):
        return weights @ x.transform(jnp.array([t, 0.0, 0.0])).array

    expected = np.array(
        [
            np.asarray(weights) @ rep.X[0] @ scipy.linalg.expm(t * rep.X[0]) @ x.array
            for t in np.asarray(angles)
        ]
    )
    individual = jax.jit(jax.vmap(jax.grad(loss)))(angles)
    joint = jax.jit(jax.grad(lambda ts: jax.vmap(loss)(ts).sum()))(angles)
    np.testing.assert_allclose(individual, expected, atol=1e-5)
    np.testing.assert_allclose(joint, expected, atol=1e-5)
