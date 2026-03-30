"""Tests for KubeEvaluator — direct kubectl-based TPU evaluation."""

import json
from unittest.mock import AsyncMock

import pytest

from kernel_evolve.evaluator import BatchEvalRequest, BatchEvalResult, EvalRequest, EvalStatus
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


def test_job_name_sanitizes_special_chars(kube_config):
  evaluator = KubeEvaluator(kube_config)
  name = evaluator._make_job_name("v001.gen5_abc123")
  assert name.startswith("kernel-eval-")
  assert "." not in name
  assert not name.endswith("-")
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


@pytest.mark.asyncio
async def test_evaluate_downloads_artifacts(kube_config, eval_request):
  """evaluate() should attempt to download GCS artifacts when gcs_prefix is present."""
  evaluator = KubeEvaluator(kube_config)
  result_json = json.dumps(
    {
      "status": "SUCCESS",
      "fitness": 2.5,
      "latency_ms": 0.5,
      "speedup": 2.5,
      "metadata": {"artifacts_gcs_prefix": "gs://glaucis-profiles/test-job"},
    }
  )

  evaluator._create_configmap = AsyncMock()
  evaluator._render_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="Complete")
  evaluator._read_logs = AsyncMock(return_value=f"EVAL_RESULT:{result_json}")
  evaluator._download_artifacts = AsyncMock()
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate(eval_request)
  assert result.status == EvalStatus.SUCCESS
  evaluator._download_artifacts.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_skips_download_when_no_gcs_prefix(kube_config, eval_request):
  """evaluate() should not call _download_artifacts when no gcs_prefix."""
  evaluator = KubeEvaluator(kube_config)
  result_json = json.dumps(
    {
      "status": "SUCCESS",
      "fitness": 2.5,
      "latency_ms": 0.5,
      "speedup": 2.5,
    }
  )

  evaluator._create_configmap = AsyncMock()
  evaluator._render_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="Complete")
  evaluator._read_logs = AsyncMock(return_value=f"EVAL_RESULT:{result_json}")
  evaluator._download_artifacts = AsyncMock()
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate(eval_request)
  assert result.status == EvalStatus.SUCCESS
  evaluator._download_artifacts.assert_not_called()


@pytest.fixture
def batch_eval_request():
  return BatchEvalRequest(
    reference_code="def ref(): pass",
    shapes=[{"M": 1024}],
    variants=[
      {"variant_id": "v1-tiling", "kernel_code": "def k1(): pass"},
      {"variant_id": "v2-pipeline", "kernel_code": "def k2(): pass"},
    ],
    rtol=1e-2,
    atol=1.0,
  )


@pytest.mark.asyncio
async def test_evaluate_batch_success(kube_config, batch_eval_request):
  evaluator = KubeEvaluator(kube_config)
  r1_json = json.dumps(
    {
      "variant_id": "v1-tiling",
      "status": "SUCCESS",
      "fitness": 1.5,
      "latency_ms": 0.5,
      "speedup": 1.5,
    }
  )
  r2_json = json.dumps(
    {
      "variant_id": "v2-pipeline",
      "status": "SUCCESS",
      "fitness": 1.2,
      "latency_ms": 0.8,
      "speedup": 1.2,
    }
  )
  logs = f"setup...\nEVAL_RESULT:{r1_json}\nEVAL_RESULT:{r2_json}\ndone"

  evaluator._create_configmap = AsyncMock()
  evaluator._render_batch_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="Complete")
  evaluator._read_logs = AsyncMock(return_value=logs)
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate_batch(batch_eval_request)
  assert isinstance(result, BatchEvalResult)
  assert len(result.results) == 2
  assert result.results["v1-tiling"].speedup == 1.5
  assert result.results["v2-pipeline"].speedup == 1.2
  evaluator._cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_batch_partial_failure(kube_config, batch_eval_request):
  evaluator = KubeEvaluator(kube_config)
  r1_json = json.dumps(
    {
      "variant_id": "v1-tiling",
      "status": "SUCCESS",
      "fitness": 1.5,
      "latency_ms": 0.5,
      "speedup": 1.5,
    }
  )
  r2_json = json.dumps(
    {
      "variant_id": "v2-pipeline",
      "status": "COMPILE_ERROR",
      "error": "bad code",
    }
  )
  logs = f"EVAL_RESULT:{r1_json}\nEVAL_RESULT:{r2_json}\n"

  evaluator._create_configmap = AsyncMock()
  evaluator._render_batch_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="Complete")
  evaluator._read_logs = AsyncMock(return_value=logs)
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate_batch(batch_eval_request)
  assert result.results["v1-tiling"].status == EvalStatus.SUCCESS
  assert result.results["v2-pipeline"].status == EvalStatus.COMPILE_ERROR
  ranked = result.ranked()
  assert len(ranked) == 1
  assert ranked[0][0] == "v1-tiling"


@pytest.mark.asyncio
async def test_evaluate_batch_job_timeout(kube_config, batch_eval_request):
  evaluator = KubeEvaluator(kube_config)
  evaluator._create_configmap = AsyncMock()
  evaluator._render_batch_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="timed_out")
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate_batch(batch_eval_request)
  assert len(result.results) == 2
  for r in result.results.values():
    assert r.status == EvalStatus.COMPILE_ERROR
    assert "timed" in r.error.lower()


@pytest.mark.asyncio
async def test_evaluate_batch_downloads_artifacts(kube_config, batch_eval_request):
  evaluator = KubeEvaluator(kube_config)
  r1_json = json.dumps(
    {
      "variant_id": "v1-tiling",
      "status": "SUCCESS",
      "fitness": 1.5,
      "latency_ms": 0.5,
      "speedup": 1.5,
      "metadata": {"artifacts_gcs_prefix": "gs://glaucis-profiles/batch-job/v1-tiling"},
    }
  )
  r2_json = json.dumps(
    {
      "variant_id": "v2-pipeline",
      "status": "SUCCESS",
      "fitness": 1.2,
      "latency_ms": 0.8,
      "speedup": 1.2,
    }
  )
  logs = f"EVAL_RESULT:{r1_json}\nEVAL_RESULT:{r2_json}\n"

  evaluator._create_configmap = AsyncMock()
  evaluator._render_batch_job_yaml = lambda *_: "rendered"
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="Complete")
  evaluator._read_logs = AsyncMock(return_value=logs)
  evaluator._download_artifacts = AsyncMock()
  evaluator._cleanup = AsyncMock()

  await evaluator.evaluate_batch(batch_eval_request)
  evaluator._download_artifacts.assert_called_once()


def test_render_batch_job_yaml_includes_deadline(kube_config, batch_eval_request, tmp_path):
  template = tmp_path / "job.yaml"
  template.write_text("deadline: ${ACTIVE_DEADLINE}\nname: ${JOB_NAME}\n")
  kube_config.job_template = str(template)
  evaluator = KubeEvaluator(kube_config)
  rendered = evaluator._render_batch_job_yaml("my-batch", batch_eval_request)
  # 2 variants * 300 + 300 = 900
  assert "deadline: 900" in rendered
