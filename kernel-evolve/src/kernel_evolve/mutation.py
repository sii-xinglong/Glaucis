"""Mutation engine: extract/inject evolve blocks and validate syntax."""

from __future__ import annotations

import ast

DEFAULT_START_MARKER = "# EVOLVE-BLOCK-START"
DEFAULT_END_MARKER = "# EVOLVE-BLOCK-END"


def extract_evolve_block(
  code: str,
  start_marker: str = DEFAULT_START_MARKER,
  end_marker: str = DEFAULT_END_MARKER,
) -> str:
  lines = code.split("\n")
  start_idx = None
  end_idx = None
  for i, line in enumerate(lines):
    if start_marker in line:
      start_idx = i
    if end_marker in line:
      end_idx = i
  if start_idx is None:
    raise ValueError(f"Could not find start marker: {start_marker}")
  if end_idx is None:
    raise ValueError(f"Could not find end marker: {end_marker}")
  return "\n".join(lines[start_idx + 1 : end_idx])


def inject_evolve_block(
  template: str,
  new_block: str,
  start_marker: str = DEFAULT_START_MARKER,
  end_marker: str = DEFAULT_END_MARKER,
) -> str:
  lines = template.split("\n")
  start_idx = None
  end_idx = None
  for i, line in enumerate(lines):
    if start_marker in line:
      start_idx = i
    if end_marker in line:
      end_idx = i
  if start_idx is None:
    raise ValueError(f"Could not find start marker: {start_marker}")
  if end_idx is None:
    raise ValueError(f"Could not find end marker: {end_marker}")
  before = lines[: start_idx + 1]
  after = lines[end_idx:]
  new_lines = new_block.split("\n")
  return "\n".join(before + new_lines + after)


def validate_syntax(code: str) -> bool:
  try:
    ast.parse(code)
    return True
  except SyntaxError:
    return False
