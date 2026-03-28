# KubeEvaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `KubeEvaluator` that submits K8s Jobs directly to the GKE TPU cluster via kubectl, enabling the kernel-evolve optimizer to run the matmul example end-to-end with Claude as the LLM backend.

**Architecture:** `KubeEvaluator` implements the `Evaluator` interface. It creates a ConfigMap with the eval payload, renders the existing `kernel-eval-job.yaml` template, applies it via kubectl, polls for completion, reads logs to extract `EVAL_RESULT:`, and cleans up. The CLI routes to `KubeEvaluator` or `CIDispatcher` based on config.

**Tech Stack:** Python 3.10+, asyncio, kubectl, Pydantic, Click, pytest, pytest-asyncio

---

### Task 1: Add KubeConfig and Evaluator Enum to Config

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/config.py:38-72`
- Test: `kernel-evolve/tests/test_config.py`

**Step 1: Write the failing test**

Add to `kernel-evolve/tests/test_config.py`:

```python
def test_config_with_kube_evaluator():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    llm={"provider": "anthropic", "model": "claude-opus-4-6"},
    tpu={"cluster": "c", "zone": "z", "tpu_type": "v4-8", "image": "img"},
    evaluator={
      "type": "kube",
      "namespace": "default",
      "job_template": ".github/ci/kernel-eval-job.yaml",
      "repo": "sii-xinglong/Glaucis",
      "branch": "main",
      "poll_interval": 15,
      "timeout": 600,
    },
  )
  assert cfg.evaluator.type.value == "kube"
  assert cfg.evaluator.namespace == "default"
  assert cfg.evaluator.poll_interval == 15


def test_config_evaluator_defaults():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    llm={"provider": "anthropic", "model": "claude-opus-4-6"},
    tpu={"cluster": "c", "zone": "z", "tpu_type": "v4-8", "image": "img"},
  )
  assert cfg.evaluator.type.value == "kube"
  assert cfg.evaluator.poll_interval == 15
  assert cfg.evaluator.timeout == 600
```

**Step 2: Run test to verify it fails**

Run: `cd kernel-evolve && python -m pytest tests/test_config.py::test_config_with_kube_evaluator tests/test_config.py::test_config_evaluator_defaults -v`
Expected: FAIL — `EvolveConfig` has no `evaluator` field

**Step 3: Write minimal implementation**

In `kernel-evolve/src/kernel_evolve/config.py`, add after `LLMProvider` enum (line 42):

```python
class EvaluatorType(str, Enum):
  kube = "kube"
  ci = "ci"


class EvaluatorConfig(BaseModel):
  type: EvaluatorType = EvaluatorType.kube
  namespace: str = "default"
  job_template: str = ".github/ci/kernel-eval-job.yaml"
  repo: str = ""
  branch: str = "main"
  poll_interval: int = 15
  timeout: int = 600
```

Add to `EvolveConfig`:

```python
evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
```

**Step 4: Run test to verify it passes**

Run: `cd kernel-evolve && python -m pytest tests/test_config.py -v`
Expected: ALL PASS (including existing tests)

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/config.py kernel-evolve/tests/test_config.py
git commit -m "feat(config): add EvaluatorConfig with kube/ci types"
```

---

### Task 2: Implement KubeEvaluator

**Files:**
- Create: `kernel-evolve/src/kernel_evolve/kube_evaluator.py`
- Test: `kernel-evolve/tests/test_kube_evaluator.py`

**Step 1: Write the failing tests**

Create `kernel-evolve/tests/test_kube_evaluator.py`:

```python
"""Tests for KubeEvaluator — direct kubectl-based TPU evaluation."""

import json
from unittest.mock import AsyncMock, patch

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
  template.write_text(
    "name: ${JOB_NAME}\nrepo: ${REPO}\nbranch: ${BRANCH}\nvariant: ${VARIANT_ID}\n"
  )
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
  evaluator._apply_job = AsyncMock()
  evaluator._poll_job = AsyncMock(return_value="Complete")
  evaluator._read_logs = AsyncMock(return_value="just some output with no eval result")
  evaluator._cleanup = AsyncMock()

  result = await evaluator.evaluate(eval_request)
  assert result.status == EvalStatus.COMPILE_ERROR
  assert "no result" in result.error.lower()
```

**Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_kube_evaluator.py -v`
Expected: FAIL — `kernel_evolve.kube_evaluator` does not exist

**Step 3: Write the implementation**

Create `kernel-evolve/src/kernel_evolve/kube_evaluator.py`:

```python
"""Direct kubectl-based TPU kernel evaluation via K8s Jobs."""

from __future__ import annotations

import asyncio
import json
import string
from dataclasses import dataclass, field
from pathlib import Path

from kernel_evolve.evaluator import EvalRequest, EvalResult, Evaluator


@dataclass
class KubeConfig:
  namespace: str = "default"
  job_template: str = ".github/ci/kernel-eval-job.yaml"
  repo: str = ""
  branch: str = "main"
  poll_interval: int = 15
  timeout: int = 600


class KubeEvaluator(Evaluator):
  """Evaluates kernel variants by submitting K8s Jobs directly via kubectl."""

  def __init__(self, config: KubeConfig):
    self._config = config

  def _make_job_name(self, variant_id: str) -> str:
    safe_id = variant_id.lower().replace("_", "-")
    name = f"kernel-eval-{safe_id}"
    return name[:63]

  def _render_job_yaml(self, job_name: str, request: EvalRequest) -> str:
    template_text = Path(self._config.job_template).read_text()
    mapping = {
      "JOB_NAME": job_name,
      "BRANCH": self._config.branch,
      "REPO": self._config.repo,
      "VARIANT_ID": request.variant_id,
    }
    tmpl = string.Template(template_text)
    return tmpl.safe_substitute(mapping)

  async def _run_kubectl(self, *args: str, stdin: str | None = None) -> tuple[str, str, int]:
    cmd = ["kubectl", *args]
    proc = await asyncio.create_subprocess_exec(
      *cmd,
      stdin=asyncio.subprocess.PIPE if stdin else None,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=stdin.encode() if stdin else None)
    return stdout.decode(), stderr.decode(), proc.returncode

  async def _create_configmap(self, job_name: str, payload: str) -> None:
    cm_name = f"{job_name}-payload"
    dry_run_out, _, rc = await self._run_kubectl(
      "create", "configmap", cm_name,
      f"--from-literal=payload={payload}",
      "--dry-run=client", "-o", "yaml",
      "-n", self._config.namespace,
    )
    if rc != 0:
      raise RuntimeError(f"Failed to generate ConfigMap YAML")
    _, stderr, rc = await self._run_kubectl(
      "apply", "-f", "-",
      "-n", self._config.namespace,
      stdin=dry_run_out,
    )
    if rc != 0:
      raise RuntimeError(f"Failed to apply ConfigMap: {stderr}")
    await self._run_kubectl(
      "label", "configmap", cm_name,
      "app=kernel-eval", "--overwrite",
      "-n", self._config.namespace,
    )

  async def _apply_job(self, rendered_yaml: str) -> None:
    _, stderr, rc = await self._run_kubectl(
      "apply", "-f", "-",
      "-n", self._config.namespace,
      stdin=rendered_yaml,
    )
    if rc != 0:
      raise RuntimeError(f"Failed to apply Job: {stderr}")

  async def _poll_job(self, job_name: str) -> str:
    elapsed = 0
    while elapsed < self._config.timeout:
      stdout, _, rc = await self._run_kubectl(
        "get", "job", job_name,
        "-n", self._config.namespace,
        "-o", "jsonpath={.status.conditions[*].type}",
      )
      conditions = stdout.strip()
      if "Complete" in conditions:
        return "Complete"
      if "Failed" in conditions:
        return "Failed"
      await asyncio.sleep(self._config.poll_interval)
      elapsed += self._config.poll_interval
    return "timed_out"

  async def _read_logs(self, job_name: str) -> str:
    stdout, _, _ = await self._run_kubectl(
      "logs", f"job/{job_name}",
      "-c", "kernel-eval",
      "-n", self._config.namespace,
    )
    return stdout

  async def _cleanup(self, job_name: str) -> None:
    cm_name = f"{job_name}-payload"
    await self._run_kubectl(
      "delete", "job", job_name,
      "-n", self._config.namespace,
      "--ignore-not-found",
    )
    await self._run_kubectl(
      "delete", "configmap", cm_name,
      "-n", self._config.namespace,
      "--ignore-not-found",
    )

  async def evaluate(self, request: EvalRequest) -> EvalResult:
    job_name = self._make_job_name(request.variant_id)
    payload = request.encode_b64()

    try:
      await self._create_configmap(job_name, payload)
      rendered_yaml = self._render_job_yaml(job_name, request)
      await self._apply_job(rendered_yaml)
    except Exception as e:
      await self._cleanup(job_name)
      return EvalResult.compile_error(f"Failed to submit job: {e}")

    status = await self._poll_job(job_name)

    if status not in ("Complete", "Failed"):
      await self._cleanup(job_name)
      return EvalResult.compile_error(f"Job timed out after {self._config.timeout}s")

    logs = await self._read_logs(job_name)
    await self._cleanup(job_name)

    for line in logs.split("\n"):
      if "EVAL_RESULT:" in line:
        json_str = line.split("EVAL_RESULT:", 1)[1].strip()
        return EvalResult.from_dict(json.loads(json_str))

    error_snippet = logs[-500:] if len(logs) > 500 else logs
    return EvalResult.compile_error(f"No result in job logs. Tail: {error_snippet}")
```

**Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_kube_evaluator.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/kube_evaluator.py kernel-evolve/tests/test_kube_evaluator.py
git commit -m "feat: add KubeEvaluator for direct kubectl-based TPU evaluation"
```

---

### Task 3: Update CLI to Route to KubeEvaluator

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/cli.py:49-56`
- Modify: `kernel-evolve/tests/test_cli.py`

**Step 1: Write the failing test**

Add to `kernel-evolve/tests/test_cli.py`:

```python
@pytest.fixture
def kube_config_file(tmp_path):
  output_dir = tmp_path / "run_output"
  template = tmp_path / "job.yaml"
  template.write_text("apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: ${JOB_NAME}\n")
  cfg = tmp_path / "test_kube.yaml"
  cfg.write_text(f"""\
kernel:
  name: "test"
  template: "k.py"
  reference: "r.py"
shapes:
  - {{ M: 64 }}
llm:
  provider: "anthropic"
  model: "claude-opus-4-6"
tpu:
  cluster: "tpu7x-cluster"
  zone: "us-central1"
  tpu_type: "v7x"
  image: "img"
evaluator:
  type: "kube"
  job_template: "{template}"
  repo: "sii-xinglong/Glaucis"
logging:
  output_dir: "{output_dir}"
""")
  return cfg


def test_cli_run_dry_run_kube(runner, kube_config_file):
  result = runner.invoke(main, ["run", "--config", str(kube_config_file), "--dry-run"])
  assert result.exit_code == 0
  assert "Loaded config" in result.output
```

**Step 2: Run test to verify it fails**

Run: `cd kernel-evolve && python -m pytest tests/test_cli.py::test_cli_run_dry_run_kube -v`
Expected: FAIL — config validation fails on unknown `evaluator` field (if Task 1 isn't done yet) or the CLI doesn't wire up the evaluator

**Step 3: Write the implementation**

Replace the evaluator section in `kernel-evolve/src/kernel_evolve/cli.py` (lines 49-56) with:

```python
  from kernel_evolve.llm import create_provider

  provider = create_provider(cfg.llm.provider.value, cfg.llm.model, cfg.llm.temperature)

  if cfg.evaluator.type.value == "ci":
    from kernel_evolve.ci_dispatcher import CIConfig, CIDispatcher

    ci_config = CIConfig(repo=cfg.evaluator.repo, workflow="kernel-eval.yaml")
    evaluator = CIDispatcher(ci_config)
  else:
    from kernel_evolve.kube_evaluator import KubeConfig, KubeEvaluator

    kube_config = KubeConfig(
      namespace=cfg.evaluator.namespace,
      job_template=cfg.evaluator.job_template,
      repo=cfg.evaluator.repo,
      branch=cfg.evaluator.branch,
      poll_interval=cfg.evaluator.poll_interval,
      timeout=cfg.evaluator.timeout,
    )
    evaluator = KubeEvaluator(kube_config)
```

**Step 4: Run all CLI tests**

Run: `cd kernel-evolve && python -m pytest tests/test_cli.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/cli.py kernel-evolve/tests/test_cli.py
git commit -m "feat(cli): route to KubeEvaluator or CIDispatcher based on config"
```

---

### Task 4: Update matmul.yaml with Real Cluster Settings

**Files:**
- Modify: `kernel-evolve/examples/matmul.yaml`

**Step 1: Update the config**

Replace the full contents of `kernel-evolve/examples/matmul.yaml`:

```yaml
kernel:
  name: "tiled_matmul"
  template: "kernels/matmul.py"
  reference: "kernels/matmul_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:
  - { M: 1024, N: 1024, K: 1024 }
  - { M: 2048, N: 2048, K: 2048 }

correctness:
  method: "allclose"
  rtol: 1e-2
  atol: 1.0

evolution:
  population_size: 25
  num_islands: 3
  max_generations: 50
  stagnation_limit: 10
  fitness: "speedup"

llm:
  provider: "anthropic"
  model: "claude-opus-4-6"
  temperature: 0.7

evaluator:
  type: "kube"
  namespace: "default"
  job_template: ".github/ci/kernel-eval-job.yaml"
  repo: "sii-xinglong/Glaucis"
  branch: "main"
  poll_interval: 15
  timeout: 600

tpu:
  cluster: "tpu7x-cluster"
  zone: "us-central1"
  tpu_type: "v7x"
  namespace: "default"
  image: "python:3.12"
  timeout: 300

logging:
  output_dir: "runs/matmul_001"
  perf_log: true
  charts: true
```

**Step 2: Run config test to verify it still parses**

Run: `cd kernel-evolve && python -m pytest tests/test_config.py::test_load_config_from_yaml -v`
Expected: PASS

**Step 3: Commit**

```bash
git add kernel-evolve/examples/matmul.yaml
git commit -m "chore: update matmul.yaml with real cluster settings and kube evaluator"
```

---

### Task 5: Run Full Test Suite and Verify

**Step 1: Run all tests**

Run: `cd kernel-evolve && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Run dry-run validation**

Run: `cd kernel-evolve && kernel-evolve run --config examples/matmul.yaml --dry-run`
Expected: `Loaded config: tiled_matmul (2 shapes, 50 generations)` + `Dry run complete. Config is valid.`

**Step 3: Commit any fixes if needed**

---

### Task 6: Launch the Full Evolution Run

**Prereqs:**
- `kubectl` has credentials for `tpu7x-cluster`: run `gcloud container clusters get-credentials tpu7x-cluster --region=us-central1`
- `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` env var is set

**Step 1: Start the run**

Run from the `kernel-evolve/` directory:

```bash
cd kernel-evolve && kernel-evolve run --config examples/matmul.yaml
```

Expected output pattern:
```
Loaded config: tiled_matmul (2 shapes, 50 generations)
Starting evolution: 3 islands, 25 population
```

Then it will iterate through generations, submitting K8s Jobs and reporting results.

**Step 2: Monitor progress**

In a separate terminal:
```bash
kernel-evolve status runs/matmul_001
```

Or watch K8s jobs:
```bash
kubectl get jobs -l app=kernel-eval -n default -w
```

**Step 3: After completion, inspect results**

```bash
kernel-evolve best runs/matmul_001
cat runs/matmul_001/perf_log.md
```
