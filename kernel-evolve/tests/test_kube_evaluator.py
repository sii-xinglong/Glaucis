"""Tests for KubeEvaluator — direct kubectl-based TPU evaluation."""

import json
from unittest.mock import AsyncMock

import pytest

from kernel_evolve.evaluator import EvalRequest, EvalStatus
from kernel_evolve.kube_evaluator import KubeConfig, KubeEvaluator


@pytest.fixture
def kube_config():
  return KubeConfig(
    namespace="default",
    job_template=".github/ci/kernel-eval-job.yaml",
    repo="sii-xinglong/Glaucis",
    branch="main",
    poll_interval=1,
    timeout=10,
  )


@pytest.fixture
def eval_request():
  return EvalRequest(
    variant_id="v001_abc123",
    kernel_code="def kernel_fn(): pass",
    reference_code="def ref(): pass",
    shapes=[{"M": 1024}],
    rtol=1e-2,
    atol=1.0,
  )


def test_kube_config_defaults():
  cfg = KubeConfig()
  assert cfg.namespace == "default"
  assert cfg.poll_interval == 15
  assert cfg.timeout == 600


def test_job_name_generation(kube_config):
  evaluator = KubeEvaluator(kube_config)
  name = evaluator._make_job_name("v001_abc123")
  assert name.startswith("kernel-eval-")
  assert "abc123" in name
  assert len(name) <= 63


def test_render_job_yaml(kube_config, eval_request, tmp_path):
  template = tmp_path / "job.yaml"
  template.write_text("name: ${JOB_NAME}\nrepo: ${REPO}\nbranch: ${BRANCH}\nvariant: ${VARIANT_ID}\n")
  kube_config.job_template = str(template)
  evaluator = KubeEvaluator(kube_config)
  rendered = evaluator._render_job_yaml("my-job", eval_request)
  assert "name: my-job" in rendered
  assert "repo: sii-xinglong/Glaucis" in rendered
  assert "branch: main" in rendered
  assert "variant: v001_abc123" in rendered


@pytest.mark.asyncio
async def test_evaluate_success(kube_config, eval_request):
  evaluator = KubeEvaluator(kube_config)
  result_json = json.dumps({"status": "SUCCESS", "fitness": 2.5, "latency_ms": 0.5, "speedup": 2.5})

  evaluator._create_configmap = AsyncMock()
  evaluator._render_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="Complete")
  evaluator._read_logs = AsyncMock(return_value=f"some output\nEVAL_RESULT:{result_json}\nmore output")
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate(eval_request)
  assert result.status == EvalStatus.SUCCESS
  assert result.speedup == 2.5
  evaluator._cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_job_timeout(kube_config, eval_request):
  evaluator = KubeEvaluator(kube_config)
  evaluator._create_configmap = AsyncMock()
  evaluator._render_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="timed_out")
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate(eval_request)
  assert result.status == EvalStatus.COMPILE_ERROR
  assert "timed" in result.error.lower()
  evaluator._cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_no_result_in_logs(kube_config, eval_request):
  evaluator = KubeEvaluator(kube_config)
  evaluator._create_configmap = AsyncMock()
  evaluator._render_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="Complete")
  evaluator._read_logs = AsyncMock(return_value="just some output with no eval result")
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate(eval_request)
  assert result.status == EvalStatus.COMPILE_ERROR
  assert "no result" in result.error.lower()
