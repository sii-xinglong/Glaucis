"""Tests for mutation engine: code extraction, injection, and syntax validation."""

import pytest

from kernel_evolve.mutation import extract_evolve_block, inject_evolve_block, validate_syntax

TEMPLATE_CODE = """\
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def kernel(x_ref, y_ref, o_ref):
  # EVOLVE-BLOCK-START
  x = x_ref[:]
  y = y_ref[:]
  o_ref[:] = x + y
  # EVOLVE-BLOCK-END

def run():
  return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32))(x, y)
"""


def test_extract_evolve_block():
  block = extract_evolve_block(TEMPLATE_CODE)
  assert "x = x_ref[:]" in block
  assert "o_ref[:] = x + y" in block
  assert "EVOLVE-BLOCK" not in block


def test_extract_custom_markers():
  code = "# BEGIN\nfoo\n# END\n"
  block = extract_evolve_block(code, start_marker="# BEGIN", end_marker="# END")
  assert block.strip() == "foo"


def test_extract_missing_markers():
  with pytest.raises(ValueError, match="start marker"):
    extract_evolve_block("no markers here")


def test_inject_evolve_block():
  new_block = "  x = x_ref[:]\n  y = y_ref[:]\n  o_ref[:] = x * y"
  result = inject_evolve_block(TEMPLATE_CODE, new_block)
  assert "x * y" in result
  assert "x + y" not in result
  assert "import jax" in result
  assert "def run():" in result


def test_validate_syntax_valid():
  assert validate_syntax("def f(): return 1") is True


def test_validate_syntax_invalid():
  assert validate_syntax("def f( return") is False


def test_roundtrip():
  block = extract_evolve_block(TEMPLATE_CODE)
  result = inject_evolve_block(TEMPLATE_CODE, block)
  assert "x = x_ref[:]" in result
  assert "o_ref[:] = x + y" in result
