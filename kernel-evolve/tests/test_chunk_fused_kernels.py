"""Tests for chunk_fused_kernels kernel conventions and basic correctness."""
import ast
import pathlib

import pytest

from kernel_evolve.config import load_config
from kernel_evolve.mutation import extract_evolve_block

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def test_template_syntax():
    ast.parse((EXAMPLES / "kernels/chunk_fused_kernels.py").read_text())


def test_reference_syntax():
    ast.parse((EXAMPLES / "kernels/chunk_fused_kernels_ref.py").read_text())


def test_evolve_block_extractable():
    code = (EXAMPLES / "kernels/chunk_fused_kernels.py").read_text()
    block = extract_evolve_block(code)
    assert len(block) > 100
    assert "optimized_compute" in block


def test_template_has_optimized_compute():
    ns = {}
    exec((EXAMPLES / "kernels/chunk_fused_kernels.py").read_text(), ns)
    assert callable(ns.get("optimized_compute"))


def test_reference_has_simple_compute():
    ns = {}
    exec((EXAMPLES / "kernels/chunk_fused_kernels_ref.py").read_text(), ns)
    assert callable(ns.get("simple_compute"))


def test_reference_has_reference_fn():
    ns = {}
    exec((EXAMPLES / "kernels/chunk_fused_kernels_ref.py").read_text(), ns)
    assert callable(ns.get("reference_fn"))


def test_matching_test_data():
    """Both template and ref must generate identical test data."""
    tmpl_ns = {}
    ref_ns = {}
    exec((EXAMPLES / "kernels/chunk_fused_kernels.py").read_text(), tmpl_ns)
    exec((EXAMPLES / "kernels/chunk_fused_kernels_ref.py").read_text(), ref_ns)

    import jax.numpy as jnp

    t_q, t_k, t_v, t_g = tmpl_ns["_make_test_data"](1, 128, 2, 128, 128, 64)
    r_q, r_k, r_v, r_g = ref_ns["_make_test_data"](1, 128, 2, 128, 128, 64)

    assert jnp.allclose(t_q, r_q)
    assert jnp.allclose(t_k, r_k)
    assert jnp.allclose(t_v, r_v)
    assert jnp.allclose(t_g, r_g)


def test_config_loads():
    config = load_config(EXAMPLES / "chunk_fused_kernels.yaml")
    assert config.kernel.name == "chunk_fused_kernels"
    assert len(config.shapes) >= 1


@pytest.mark.skipif(
    not __import__("jax").default_backend().upper().startswith("TPU"),
    reason="Pallas TPU kernels require TPU backend",
)
def test_reference_forward_small():
    """Reference kernel produces output on TPU (small dims)."""
    ref_ns = {}
    exec((EXAMPLES / "kernels/chunk_fused_kernels_ref.py").read_text(), ref_ns)

    import jax.numpy as jnp

    q, k, v, g_gamma = ref_ns["_make_test_data"](1, 64, 2, 128, 128, 64)
    scale = 128 ** -0.5
    o = ref_ns["chunk_gla_ref"](
        q.astype(jnp.float32), k.astype(jnp.float32),
        v.astype(jnp.float32), g_gamma, scale, 64,
    )
    assert o.shape == (1, 64, 2, 128)
    assert jnp.isfinite(o).all()
