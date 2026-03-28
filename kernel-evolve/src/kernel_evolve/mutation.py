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

  # Detect target indentation from the start marker line
  marker_line = lines[start_idx]
  target_indent = marker_line[: len(marker_line) - len(marker_line.lstrip())]

  # Find the minimum indentation in the new block (to detect the base level)
  new_lines = new_block.split("\n")
  min_indent = None
  for line in new_lines:
    stripped = line.lstrip()
    if stripped:
      line_indent = len(line) - len(stripped)
      if min_indent is None or line_indent < min_indent:
        min_indent = line_indent
  if min_indent is None:
    min_indent = 0

  # Re-indent: strip the detected base indent, then prepend the target indent
  reindented = []
  for line in new_lines:
    stripped = line.lstrip()
    if not stripped:
      reindented.append("")
    else:
      line_indent = len(line) - len(stripped)
      relative_indent = " " * (line_indent - min_indent)
      reindented.append(target_indent + relative_indent + stripped)
  return "\n".join(before + reindented + after)


def validate_syntax(code: str) -> bool:
  try:
    ast.parse(code)
    return True
  except SyntaxError:
    return False
