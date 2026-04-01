"""Tests for fused_chunk_simple_gla kernel template and reference."""
import ast
from pathlib import Path

import pytest

from kernel_evolve.mutation import extract_evolve_block
from kernel_evolve.config import load_config


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
TEMPLATE = EXAMPLES_DIR / "kernels" / "fused_chunk_simple_gla.py"
REFERENCE = EXAMPLES_DIR / "kernels" / "fused_chunk_simple_gla_ref.py"
CONFIG = EXAMPLES_DIR / "fused_chunk_simple_gla.yaml"


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
    assert config.kernel.name == "fused_chunk_simple_gla"
    assert len(config.shapes) == 2
    assert config.shapes[0]["B"] == 10
    assert config.shapes[1]["B"] == 12
    assert config.shapes[0]["T"] == 4096
