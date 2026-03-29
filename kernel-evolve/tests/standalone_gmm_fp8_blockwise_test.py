"""Standalone GMM block-wise FP8 quantization test.

Exercises the forward and backward paths of Grouped Matrix Multiply (GMM)
with DeepSeek-V3 / Transformer Engine style block-wise FP8 quantization,
using the tokamax Pallas TPU kernel and qwix quantization library.

Dependencies: jax, jaxlib, qwix, tokamax  (no ant-pretrain imports)

Model reference (al_model.yml):
  - num_experts: 256, num_experts_per_tok: 8
  - base_emb_dim: 2048, base_moe_mlp_dim: 512
  - head_dim: 128, max_target_length: 4096
  - per_device_batch_size: 2

Training reference (pretrain_al_model.sh):
  - ICI_EXPERT_PARALLELISM=8 (EP=8, each shard holds 256/8=32 experts)
  - tokens_per_device = per_device_batch_size * max_target_length = 8192
  - dispatched_per_shard ~ tokens_per_device * topk / EP = 8192

GMM shapes in real training:
  Gate/Up projection: lhs [M, 2048] @ rhs [G, 2048, 512]  -> [M, 512]
  Down projection:    lhs [M, 512]  @ rhs [G, 512, 2048]   -> [M, 2048]

Usage:
  python tests/standalone_gmm_fp8_blockwise_test.py          # run all tests
  python tests/standalone_gmm_fp8_blockwise_test.py -v        # verbose
  pytest tests/standalone_gmm_fp8_blockwise_test.py -v        # via pytest
"""

import dataclasses
import functools
import unittest
from typing import List, Tuple

import jax
import jax.numpy as jnp

import qwix
import qwix.pallas as qpl

from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as tokamax_backend


# ---------------------------------------------------------------------------
# AL Model MoE parameters (from configs/models/al_model.yml)
# ---------------------------------------------------------------------------
NUM_EXPERTS = 256           # num_experts
NUM_EXPERTS_PER_TOK = 8     # num_experts_per_tok (topk)
EMB_DIM = 2048              # base_emb_dim (hidden_size)
MOE_MLP_DIM = 512           # base_moe_mlp_dim (expert FFN hidden)
MAX_SEQ_LEN = 4096          # max_target_length
PER_DEVICE_BATCH_SIZE = 2   # per_device_batch_size

# Training parallelism (from pretrain_al_model.sh)
EP = 8                      # ICI_EXPERT_PARALLELISM

# Derived per-EP-shard dimensions (real training sizes)
TOKENS_PER_DEVICE = PER_DEVICE_BATCH_SIZE * MAX_SEQ_LEN         # 8192
EXPERTS_PER_SHARD = NUM_EXPERTS // EP                            # 32
# After topk routing + all-to-all, each EP shard processes ~this many tokens
DISPATCHED_TOKENS = TOKENS_PER_DEVICE * NUM_EXPERTS_PER_TOK // EP  # 8192

# FP8 block-wise quantization (tile_size=128, E4M3 for all)
TILE_SIZE = 128
FP8_DTYPE = jnp.float8_e4m3fn


# ---------------------------------------------------------------------------
# Block-wise QArray detection (from ops.py)
# ---------------------------------------------------------------------------
def _is_blockwise_qarray(q: qpl.QArray) -> bool:
    """Check if a QArray has block-wise (not per-channel) scales."""
    return any(
        sd != 1 and sd != qd for sd, qd in zip(q.scale.shape, q.qvalue.shape)
    )


# ---------------------------------------------------------------------------
# GMM with fp8_blockwise custom VJP (standalone re-implementation of ops.py)
# ---------------------------------------------------------------------------
def gmm_fp8_blockwise(
    lhs: jnp.ndarray,            # [M, K] activations (bf16)
    rhs: jnp.ndarray,            # [G, K, N] weights (bf16)
    group_sizes: jnp.ndarray,    # [G] int32
    tiling: tuple[int, ...] = (128, 128, 128) * 3,
    group_offset: jnp.ndarray | None = None,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
) -> jnp.ndarray:
    """GMM with fp8_blockwise quantization and tokamax backend."""
    qt_rule = qwix.QtRule(
        weight_qtype=FP8_DTYPE,
        act_qtype=FP8_DTYPE,
        bwd_qtype=FP8_DTYPE,
        tile_size=TILE_SIZE,
        weight_calibration_method="absmax",
        act_calibration_method="absmax",
        bwd_calibration_method="absmax",
    )
    fwd_bwd = lambda *a: _gmm_fwd(*a)[0]
    fwd_bwd = jax.custom_vjp(fwd_bwd, nondiff_argnums=(3, 4, 5))
    fwd_bwd.defvjp(
        _gmm_fwd,
        functools.partial(_gmm_bwd, lhs.dtype, rhs.dtype),
    )
    return fwd_bwd(lhs, rhs, group_sizes, qt_rule, tiling, weight_gather_axes)


def _gmm_fwd(
    lhs,          # [M, K]
    rhs,          # [G, K, N]
    group_sizes,  # [G]
    qt_rule,
    tiling,
    weight_gather_axes,
):
    tile_size = qt_rule.tile_size

    # Cap kernel tiling to quantization block size.
    tiling = tuple(min(t, tile_size) for t in tiling)

    # --- TE-style dual quantization from BF16 source ---
    lhs_bf16 = lhs
    # lhs [M, K] with (1, tile_size) blocks for forward gmm
    lhs = qpl.quantize(
        lhs_bf16,
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )
    # lhs_t [K, M] with (1, tile_size) blocks for backward tgmm
    lhs_t = qpl.quantize(
        lhs_bf16.swapaxes(0, 1),
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )

    # rhs [G, K, N] with (128, 128) blocks on K and N dims
    rhs = qpl.quantize(
        rhs,
        qt_rule.weight_qtype,
        channelwise_axes=[0],  # per-group only, K and N are tiled
        tiled_axes={1: tile_size, 2: tile_size},
        calibration_method=qt_rule.weight_calibration_method,
        scale_dtype=jnp.float32,
    )

    # weight gather for expert parallelism (if applicable)
    if weight_gather_axes:
        for axis_name, axis_idx in weight_gather_axes:
            rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
            rhs_scale = jax.lax.all_gather(rhs.scale, axis_name, axis=axis_idx, tiled=True)
            rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue, scale=rhs_scale)

    out = tokamax_backend.gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=jnp.float32,
        tiling=tiling[:3],
        group_offset=None,
        transpose_rhs=False,
        interpret=False,
    )
    return out, (lhs, rhs, group_sizes, lhs_t)


def _gmm_bwd(
    lhs_dtype,
    rhs_dtype,
    qt_rule,
    tiling,
    weight_gather_axes,
    residual,
    grad,
):
    lhs, rhs, group_sizes, lhs_t = residual
    num_actual_groups = rhs.shape[0]
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    dlhs_dout = grad
    drhs_dout = grad

    # Block-wise: keep QArrays as-is, kernel handles scale internally.
    # (No per-channel dual quantization trick for blockwise mode.)

    # Quantize gradients with TE-style 1x128 blocks
    # dlhs_dout [M, N]: channelwise on M, tiled on N
    dlhs_dout = qpl.quantize(
        dlhs_dout,
        qt_rule.bwd_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )
    # drhs_dout [M, N]: channelwise on N, tiled on M
    #   tgmm reduction_axis=0, eps=tile_size >= min_addressable_size=16
    drhs_dout = qpl.quantize(
        drhs_dout,
        qt_rule.bwd_qtype,
        channelwise_axes=[1],
        tiled_axes={0: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )

    # dlhs = dout @ rhs^T
    dlhs = tokamax_backend.gmm(
        lhs=dlhs_dout,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=lhs_dtype,
        tiling=tiling[3:6] if len(tiling) >= 6 else tiling[:3],
        group_offset=None,
        transpose_rhs=True,   # backward: transpose weight
        interpret=False,
    )

    # drhs = lhs^T @ dout  (using pre-computed lhs_t)
    drhs = tokamax_backend.tgmm(
        lhs=lhs_t,            # [K, M] pre-quantized from forward
        rhs=drhs_dout,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=rhs_dtype,
        tiling=tiling[6:9] if len(tiling) >= 9 else tiling[:3],
        group_offset=None,
        num_actual_groups=num_actual_groups,
        interpret=False,
    )

    return dlhs, drhs, None


# ---------------------------------------------------------------------------
# Reference BF16 GMM (no quantization) for correctness comparison
# ---------------------------------------------------------------------------
def gmm_bf16_reference(
    lhs: jnp.ndarray,         # [M, K]
    rhs: jnp.ndarray,         # [G, K, N]
    group_sizes: jnp.ndarray, # [G]
) -> jnp.ndarray:
    """Reference GMM in BF16 using tokamax backend (no quantization)."""
    return tokamax_backend.gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=jnp.float32,
        tiling=(128, 128, 128),
        transpose_rhs=False,
        interpret=False,
    )


# ---------------------------------------------------------------------------
# Helper: create test data for a given projection shape
# ---------------------------------------------------------------------------
def _make_test_data(M, K, N, G, seed=42):
    """Create (lhs, rhs, group_sizes) for a GMM test case."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    lhs = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (G, K, N), dtype=jnp.bfloat16)
    # Distribute M tokens evenly across G groups (all must be multiples of 128
    # for tgmm tiling compatibility).
    base = (M // G // TILE_SIZE) * TILE_SIZE
    assert base > 0, f"M={M} / G={G} = {M // G} must be >= {TILE_SIZE}"
    sizes = [base] * G
    # Distribute remainder to the last group (still a multiple of 128 because
    # M and base*G are both multiples of 128 when M is a multiple of G*128).
    sizes[-1] = M - base * (G - 1)
    group_sizes = jnp.array(sizes, dtype=jnp.int32)
    return lhs, rhs, group_sizes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestQArrayShapes(unittest.TestCase):
    """Verify QArray shapes match the block-wise FP8 spec at real dimensions."""

    def test_gate_up_activation_shape(self):
        """Gate/Up activation [8192, 2048] -> qvalue same, scale [8192, 16]."""
        lhs = jax.random.normal(jax.random.PRNGKey(0), (DISPATCHED_TOKENS, EMB_DIM), dtype=jnp.bfloat16)
        q = qpl.quantize(lhs, FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE}, scale_dtype=jnp.float32)
        self.assertEqual(q.qvalue.shape, (DISPATCHED_TOKENS, EMB_DIM))
        self.assertEqual(q.qvalue.dtype, FP8_DTYPE)
        self.assertEqual(q.scale.shape, (DISPATCHED_TOKENS, EMB_DIM // TILE_SIZE))  # (8192, 16)

    def test_gate_up_activation_transposed_shape(self):
        """Transposed activation [2048, 8192] -> scale [2048, 64]."""
        lhs = jax.random.normal(jax.random.PRNGKey(0), (DISPATCHED_TOKENS, EMB_DIM), dtype=jnp.bfloat16)
        q = qpl.quantize(lhs.swapaxes(0, 1), FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE}, scale_dtype=jnp.float32)
        self.assertEqual(q.qvalue.shape, (EMB_DIM, DISPATCHED_TOKENS))
        self.assertEqual(q.scale.shape, (EMB_DIM, DISPATCHED_TOKENS // TILE_SIZE))  # (2048, 64)

    def test_gate_up_weight_shape(self):
        """Gate/Up weight [32, 2048, 512] -> scale [32, 16, 4]."""
        rhs = jax.random.normal(jax.random.PRNGKey(1), (EXPERTS_PER_SHARD, EMB_DIM, MOE_MLP_DIM), dtype=jnp.bfloat16)
        q = qpl.quantize(rhs, FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE, 2: TILE_SIZE}, scale_dtype=jnp.float32)
        self.assertEqual(q.qvalue.shape, (EXPERTS_PER_SHARD, EMB_DIM, MOE_MLP_DIM))
        self.assertEqual(q.scale.shape, (EXPERTS_PER_SHARD, EMB_DIM // TILE_SIZE, MOE_MLP_DIM // TILE_SIZE))  # (32, 16, 4)

    def test_down_weight_shape(self):
        """Down weight [32, 512, 2048] -> scale [32, 4, 16]."""
        rhs = jax.random.normal(jax.random.PRNGKey(2), (EXPERTS_PER_SHARD, MOE_MLP_DIM, EMB_DIM), dtype=jnp.bfloat16)
        q = qpl.quantize(rhs, FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE, 2: TILE_SIZE}, scale_dtype=jnp.float32)
        self.assertEqual(q.scale.shape, (EXPERTS_PER_SHARD, MOE_MLP_DIM // TILE_SIZE, EMB_DIM // TILE_SIZE))  # (32, 4, 16)

    def test_gradient_dlhs_dout_shape(self):
        """Gradient dout [8192, 512] channelwise=[0] tiled={1:128} -> scale [8192, 4]."""
        dout = jax.random.normal(jax.random.PRNGKey(3), (DISPATCHED_TOKENS, MOE_MLP_DIM), dtype=jnp.bfloat16)
        q = qpl.quantize(dout, FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE}, scale_dtype=jnp.float32)
        self.assertEqual(q.scale.shape, (DISPATCHED_TOKENS, MOE_MLP_DIM // TILE_SIZE))  # (8192, 4)

    def test_gradient_drhs_dout_shape(self):
        """Gradient dout [8192, 512] channelwise=[1] tiled={0:128} -> scale [64, 512]."""
        dout = jax.random.normal(jax.random.PRNGKey(4), (DISPATCHED_TOKENS, MOE_MLP_DIM), dtype=jnp.bfloat16)
        q = qpl.quantize(dout, FP8_DTYPE, channelwise_axes=[1], tiled_axes={0: TILE_SIZE}, scale_dtype=jnp.float32)
        self.assertEqual(q.scale.shape, (DISPATCHED_TOKENS // TILE_SIZE, MOE_MLP_DIM))  # (64, 512)

    def test_is_blockwise_qarray(self):
        """Block-wise QArrays are correctly identified."""
        lhs = jax.random.normal(jax.random.PRNGKey(5), (DISPATCHED_TOKENS, EMB_DIM), dtype=jnp.bfloat16)

        q_block = qpl.quantize(lhs, FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE}, scale_dtype=jnp.float32)
        self.assertTrue(_is_blockwise_qarray(q_block))

        q_channel = qpl.quantize(lhs, FP8_DTYPE, channelwise_axes=[0], scale_dtype=jnp.float32)
        self.assertFalse(_is_blockwise_qarray(q_channel))


class TestGmmFp8BlockwiseGateUp(unittest.TestCase):
    """Gate/Up projection: [8192, 2048] @ [32, 2048, 512] -> [8192, 512].

    Real per-EP-shard dimensions from AL model training:
      M = 8192  (tokens_per_device * topk / EP)
      K = 2048  (base_emb_dim)
      N = 512   (base_moe_mlp_dim)
      G = 32    (num_experts / EP)
    """

    M, K, N, G = DISPATCHED_TOKENS, EMB_DIM, MOE_MLP_DIM, EXPERTS_PER_SHARD

    def setUp(self):
        self.lhs, self.rhs, self.group_sizes = _make_test_data(self.M, self.K, self.N, self.G, seed=42)

    def test_forward_shape_and_dtype(self):
        out = gmm_fp8_blockwise(self.lhs, self.rhs, self.group_sizes)
        self.assertEqual(out.shape, (self.M, self.N))
        self.assertEqual(out.dtype, jnp.float32)

    def test_forward_finite(self):
        out = gmm_fp8_blockwise(self.lhs, self.rhs, self.group_sizes)
        self.assertTrue(jnp.all(jnp.isfinite(out)))

    def test_forward_close_to_bf16(self):
        out_fp8 = gmm_fp8_blockwise(self.lhs, self.rhs, self.group_sizes)
        out_ref = gmm_bf16_reference(self.lhs, self.rhs, self.group_sizes)
        rel_err = float(jnp.linalg.norm(out_fp8 - out_ref) / (jnp.linalg.norm(out_ref) + 1e-6))
        print(f"\n  Gate/Up forward relative error: {rel_err:.6f}")
        self.assertLess(rel_err, 0.15)

    def test_backward_shapes(self):
        def loss_fn(lhs, rhs):
            return gmm_fp8_blockwise(lhs, rhs, self.group_sizes).sum()

        g_lhs, g_rhs = jax.grad(loss_fn, argnums=(0, 1))(self.lhs, self.rhs)
        self.assertEqual(g_lhs.shape, (self.M, self.K))
        self.assertEqual(g_rhs.shape, (self.G, self.K, self.N))

    def test_backward_finite(self):
        def loss_fn(lhs, rhs):
            return gmm_fp8_blockwise(lhs, rhs, self.group_sizes).sum()

        g_lhs, g_rhs = jax.grad(loss_fn, argnums=(0, 1))(self.lhs, self.rhs)
        self.assertTrue(jnp.all(jnp.isfinite(g_lhs)), "grad_lhs has non-finite values")
        self.assertTrue(jnp.all(jnp.isfinite(g_rhs)), "grad_rhs has non-finite values")

    def test_backward_close_to_bf16(self):
        def loss_fp8(lhs, rhs):
            return gmm_fp8_blockwise(lhs, rhs, self.group_sizes).sum()

        def loss_ref(lhs, rhs):
            return gmm_bf16_reference(lhs, rhs, self.group_sizes).sum()

        g_fp8 = jax.grad(loss_fp8, argnums=(0, 1))(self.lhs, self.rhs)
        g_ref = jax.grad(loss_ref, argnums=(0, 1))(self.lhs, self.rhs)

        for name, gf, gr in [("grad_lhs", g_fp8[0], g_ref[0]), ("grad_rhs", g_fp8[1], g_ref[1])]:
            rel_err = float(jnp.linalg.norm((gf - gr).astype(jnp.float32)) / (jnp.linalg.norm(gr.astype(jnp.float32)) + 1e-6))
            print(f"  Gate/Up backward {name} relative error: {rel_err:.6f}")
            self.assertLess(rel_err, 0.5, f"{name} too far from BF16: {rel_err:.4f}")


class TestGmmFp8BlockwiseDown(unittest.TestCase):
    """Down projection: [8192, 512] @ [32, 512, 2048] -> [8192, 2048].

    Real per-EP-shard dimensions:
      M = 8192, K = 512 (moe_mlp_dim), N = 2048 (emb_dim), G = 32
    """

    M, K, N, G = DISPATCHED_TOKENS, MOE_MLP_DIM, EMB_DIM, EXPERTS_PER_SHARD

    def setUp(self):
        self.lhs, self.rhs, self.group_sizes = _make_test_data(self.M, self.K, self.N, self.G, seed=99)

    def test_forward_shape_and_dtype(self):
        out = gmm_fp8_blockwise(self.lhs, self.rhs, self.group_sizes)
        self.assertEqual(out.shape, (self.M, self.N))
        self.assertEqual(out.dtype, jnp.float32)

    def test_forward_finite(self):
        out = gmm_fp8_blockwise(self.lhs, self.rhs, self.group_sizes)
        self.assertTrue(jnp.all(jnp.isfinite(out)))

    def test_forward_close_to_bf16(self):
        out_fp8 = gmm_fp8_blockwise(self.lhs, self.rhs, self.group_sizes)
        out_ref = gmm_bf16_reference(self.lhs, self.rhs, self.group_sizes)
        rel_err = float(jnp.linalg.norm(out_fp8 - out_ref) / (jnp.linalg.norm(out_ref) + 1e-6))
        print(f"\n  Down forward relative error: {rel_err:.6f}")
        self.assertLess(rel_err, 0.15)

    def test_backward_shapes(self):
        def loss_fn(lhs, rhs):
            return gmm_fp8_blockwise(lhs, rhs, self.group_sizes).sum()

        g_lhs, g_rhs = jax.grad(loss_fn, argnums=(0, 1))(self.lhs, self.rhs)
        self.assertEqual(g_lhs.shape, (self.M, self.K))
        self.assertEqual(g_rhs.shape, (self.G, self.K, self.N))

    def test_backward_finite(self):
        def loss_fn(lhs, rhs):
            return gmm_fp8_blockwise(lhs, rhs, self.group_sizes).sum()

        g_lhs, g_rhs = jax.grad(loss_fn, argnums=(0, 1))(self.lhs, self.rhs)
        self.assertTrue(jnp.all(jnp.isfinite(g_lhs)))
        self.assertTrue(jnp.all(jnp.isfinite(g_rhs)))

    def test_backward_close_to_bf16(self):
        def loss_fp8(lhs, rhs):
            return gmm_fp8_blockwise(lhs, rhs, self.group_sizes).sum()

        def loss_ref(lhs, rhs):
            return gmm_bf16_reference(lhs, rhs, self.group_sizes).sum()

        g_fp8 = jax.grad(loss_fp8, argnums=(0, 1))(self.lhs, self.rhs)
        g_ref = jax.grad(loss_ref, argnums=(0, 1))(self.lhs, self.rhs)

        for name, gf, gr in [("grad_lhs", g_fp8[0], g_ref[0]), ("grad_rhs", g_fp8[1], g_ref[1])]:
            rel_err = float(jnp.linalg.norm((gf - gr).astype(jnp.float32)) / (jnp.linalg.norm(gr.astype(jnp.float32)) + 1e-6))
            print(f"  Down backward {name} relative error: {rel_err:.6f}")
            self.assertLess(rel_err, 0.5, f"{name} too far from BF16: {rel_err:.4f}")


class TestGmmFp8BlockwiseFullModel(unittest.TestCase):
    """Full model (no EP): [65536, 2048] @ [256, 2048, 512] -> [65536, 512].

    Real dimensions without expert parallelism:
      M = tokens_per_device * topk = 8192 * 8 = 65536
      K = 2048, N = 512, G = 256
    """

    M = TOKENS_PER_DEVICE * NUM_EXPERTS_PER_TOK   # 65536
    K, N, G = EMB_DIM, MOE_MLP_DIM, NUM_EXPERTS   # 2048, 512, 256

    def setUp(self):
        self.lhs, self.rhs, self.group_sizes = _make_test_data(self.M, self.K, self.N, self.G, seed=7)

    def test_forward_shape_and_finite(self):
        out = gmm_fp8_blockwise(self.lhs, self.rhs, self.group_sizes)
        self.assertEqual(out.shape, (self.M, self.N))
        self.assertTrue(jnp.all(jnp.isfinite(out)))

    def test_backward_shapes_and_finite(self):
        def loss_fn(lhs, rhs):
            return gmm_fp8_blockwise(lhs, rhs, self.group_sizes).sum()

        g_lhs, g_rhs = jax.grad(loss_fn, argnums=(0, 1))(self.lhs, self.rhs)
        self.assertEqual(g_lhs.shape, (self.M, self.K))
        self.assertEqual(g_rhs.shape, (self.G, self.K, self.N))
        self.assertTrue(jnp.all(jnp.isfinite(g_lhs)))
        self.assertTrue(jnp.all(jnp.isfinite(g_rhs)))


class TestTokamaxBackend(unittest.TestCase):
    """Verify tokamax gmm/tgmm accept block-wise FP8 QArrays at real dims."""

    def test_gmm_accepts_qarray_gate_up(self):
        """tokamax.gmm: QArray [8192, 2048] @ QArray [32, 2048, 512]."""
        M, K, N, G = DISPATCHED_TOKENS, EMB_DIM, MOE_MLP_DIM, EXPERTS_PER_SHARD
        k1, k2 = jax.random.split(jax.random.PRNGKey(10))

        lhs_q = qpl.quantize(
            jax.random.normal(k1, (M, K), dtype=jnp.bfloat16),
            FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE}, scale_dtype=jnp.float32,
        )
        rhs_q = qpl.quantize(
            jax.random.normal(k2, (G, K, N), dtype=jnp.bfloat16),
            FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE, 2: TILE_SIZE}, scale_dtype=jnp.float32,
        )
        out = tokamax_backend.gmm(
            lhs=lhs_q, rhs=rhs_q, group_sizes=jnp.full(G, M // G, dtype=jnp.int32),
            precision=jax.lax.Precision.DEFAULT, out_dtype=jnp.float32,
            tiling=(TILE_SIZE, TILE_SIZE, TILE_SIZE), transpose_rhs=False, interpret=False,
        )
        self.assertEqual(out.shape, (M, N))
        self.assertTrue(jnp.all(jnp.isfinite(out)))

    def test_tgmm_accepts_qarray_gate_up(self):
        """tokamax.tgmm: QArray [2048, 8192] @ QArray [8192, 512] -> [32, 2048, 512]."""
        M, K, N, G = DISPATCHED_TOKENS, EMB_DIM, MOE_MLP_DIM, EXPERTS_PER_SHARD
        k1, k2 = jax.random.split(jax.random.PRNGKey(11))

        lhs_t_q = qpl.quantize(
            jax.random.normal(k1, (K, M), dtype=jnp.bfloat16),  # [K, M]
            FP8_DTYPE, channelwise_axes=[0], tiled_axes={1: TILE_SIZE}, scale_dtype=jnp.float32,
        )
        rhs_q = qpl.quantize(
            jax.random.normal(k2, (M, N), dtype=jnp.bfloat16),
            FP8_DTYPE, channelwise_axes=[1], tiled_axes={0: TILE_SIZE}, scale_dtype=jnp.float32,
        )
        out = tokamax_backend.tgmm(
            lhs=lhs_t_q, rhs=rhs_q, group_sizes=jnp.full(G, M // G, dtype=jnp.int32),
            precision=jax.lax.Precision.DEFAULT, out_dtype=jnp.bfloat16,
            tiling=(TILE_SIZE, TILE_SIZE, TILE_SIZE), num_actual_groups=G, interpret=False,
        )
        self.assertEqual(out.shape, (G, K, N))
        self.assertTrue(jnp.all(jnp.isfinite(out)))


class TestQtRuleConfig(unittest.TestCase):
    """Verify QtRule configuration matches the fp8_blockwise spec."""

    def test_qt_rule_fields(self):
        rule = qwix.QtRule(
            module_path="decoder/.*layers.*",
            weight_qtype=FP8_DTYPE,
            act_qtype=FP8_DTYPE,
            bwd_qtype=FP8_DTYPE,
            tile_size=TILE_SIZE,
            op_names=("gmm",),
        )
        self.assertEqual(rule.tile_size, 128)
        self.assertEqual(rule.act_qtype, jnp.float8_e4m3fn)
        self.assertEqual(rule.weight_qtype, jnp.float8_e4m3fn)
        self.assertEqual(rule.bwd_qtype, jnp.float8_e4m3fn)
        self.assertIn("gmm", rule.op_names)

    def test_qt_rule_provider(self):
        rule = qwix.QtRule(
            weight_qtype=FP8_DTYPE, act_qtype=FP8_DTYPE, bwd_qtype=FP8_DTYPE,
            tile_size=TILE_SIZE, op_names=("gmm",),
        )
        provider = qwix.QtProvider([rule])
        self.assertIsNotNone(provider)


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"AL model: experts={NUM_EXPERTS}, topk={NUM_EXPERTS_PER_TOK}, "
          f"emb_dim={EMB_DIM}, moe_mlp_dim={MOE_MLP_DIM}, EP={EP}")
    print(f"Per-EP-shard: M={DISPATCHED_TOKENS}, G={EXPERTS_PER_SHARD}")
    print(f"Full model: M={TOKENS_PER_DEVICE * NUM_EXPERTS_PER_TOK}, G={NUM_EXPERTS}")
    print(f"Gate/Up: [{DISPATCHED_TOKENS}, {EMB_DIM}] @ [{EXPERTS_PER_SHARD}, {EMB_DIM}, {MOE_MLP_DIM}]")
    print(f"Down:    [{DISPATCHED_TOKENS}, {MOE_MLP_DIM}] @ [{EXPERTS_PER_SHARD}, {MOE_MLP_DIM}, {EMB_DIM}]")
    print()
    unittest.main()
