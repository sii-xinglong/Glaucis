"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
from click.testing import CliRunner

from kernel_evolve.cli import main


@pytest.fixture
def runner():
  return CliRunner()


@pytest.fixture
def config_file(tmp_path):
  output_dir = tmp_path / "run_output"
  cfg = tmp_path / "test.yaml"
  cfg.write_text(f"""\
kernel:
  name: "test"
  template: "k.py"
  reference: "r.py"
shapes:
  - {{ M: 64 }}
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
tpu:
  cluster: "c"
  zone: "z"
  tpu_type: "v4-8"
  image: "img"
logging:
  output_dir: "{output_dir}"
""")
  return cfg


def test_cli_version(runner):
  result = runner.invoke(main, ["--version"])
  assert result.exit_code == 0
  assert "0.1.0" in result.output


def test_cli_run_dry_run(runner, config_file):
  result = runner.invoke(main, ["run", "--config", str(config_file), "--dry-run"])
  assert result.exit_code == 0
  assert "Loaded config" in result.output


def test_cli_best_no_results(runner, tmp_path):
  run_dir = tmp_path / "empty_run"
  run_dir.mkdir()
  result = runner.invoke(main, ["best", str(run_dir)])
  assert "No best kernel found" in result.output
