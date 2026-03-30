"""Tests for chunked GLA kernel template and reference."""
import ast
from pathlib import Path

import pytest

from kernel_evolve.mutation import extract_evolve_block
from kernel_evolve.config import load_config


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
TEMPLATE = EXAMPLES_DIR / "kernels" / "chunk_gla.py"
REFERENCE = EXAMPLES_DIR / "kernels" / "chunk_gla_ref.py"
CONFIG = EXAMPLES_DIR / "chunk_gla.yaml"


def test_template_syntax():
    code = TEMPLATE.read_text()
    ast.parse(code)


def test_reference_syntax():
    code = REFERENCE.read_text()
    ast.parse(code)


def test_evolve_block_extractable():
    code = TEMPLATE.read_text()
    block = extract_evolve_block(code)
    assert len(block) > 100, f"Block too small: {len(block)} chars"
    assert "optimized_compute" in block


def test_template_has_optimized_compute():
    ns = {}
    exec(TEMPLATE.read_text(), ns)
    assert "optimized_compute" in ns
    assert callable(ns["optimized_compute"])


def test_reference_has_simple_compute():
    ns = {}
    exec(REFERENCE.read_text(), ns)
    assert "simple_compute" in ns
    assert callable(ns["simple_compute"])


def test_reference_has_reference_fn():
    ns = {}
    exec(REFERENCE.read_text(), ns)
    assert "reference_fn" in ns
    assert callable(ns["reference_fn"])


def test_matching_test_data():
    """Both kernels produce identical test data from the same seed."""
    tmpl_ns = {}
    ref_ns = {}
    exec(TEMPLATE.read_text(), tmpl_ns)
    exec(REFERENCE.read_text(), ref_ns)

    import jax.numpy as jnp

    t_q, t_k, t_v, t_g = tmpl_ns["_make_test_data"](2, 256, 4, 128, 128, 64)
    r_q, r_k, r_v, r_g = ref_ns["_make_test_data"](2, 256, 4, 128, 128, 64)

    assert jnp.allclose(t_q, r_q)
    assert jnp.allclose(t_k, r_k)
    assert jnp.allclose(t_v, r_v)
    assert jnp.allclose(t_g, r_g)


def test_config_loads():
    config = load_config(CONFIG)
    assert config.kernel.name == "chunk_gla"
    assert len(config.shapes) == 1
    assert config.shapes[0]["B"] == 2
    assert config.shapes[0]["T"] == 4096


def test_reference_forward_small():
    """Reference kernel produces output on CPU (small dims).

    Note: K must equal H for the g_cumsum broadcast to work, because
    g_gamma has shape (H,) and broadcast_to targets q.shape = (B,T,H,K).
    """
    ref_ns = {}
    exec(REFERENCE.read_text(), ref_ns)

    import jax.numpy as jnp

    H, K, V = 4, 4, 4
    q, k, v, g_gamma = ref_ns["_make_test_data"](1, 64, H, K, V, 16)
    scale = K ** -0.5
    o = ref_ns["chunk_gla_ref"](
        q.astype(jnp.float32), k.astype(jnp.float32),
        v.astype(jnp.float32), g_gamma, scale, 16
    )
    assert o.shape == (1, 64, H, V)
    assert jnp.isfinite(o).all()
