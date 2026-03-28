"""Direct kubectl-based TPU kernel evaluation via K8s Jobs."""

from __future__ import annotations

import asyncio
import json
import string
from dataclasses import dataclass
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
      "create",
      "configmap",
      cm_name,
      f"--from-literal=payload={payload}",
      "--dry-run=client",
      "-o",
      "yaml",
      "-n",
      self._config.namespace,
    )
    if rc != 0:
      raise RuntimeError("Failed to generate ConfigMap YAML")
    _, stderr, rc = await self._run_kubectl(
      "apply",
      "-f",
      "-",
      "-n",
      self._config.namespace,
      stdin=dry_run_out,
    )
    if rc != 0:
      raise RuntimeError(f"Failed to apply ConfigMap: {stderr}")
    await self._run_kubectl(
      "label",
      "configmap",
      cm_name,
      "app=kernel-eval",
      "--overwrite",
      "-n",
      self._config.namespace,
    )

  async def _apply_job(self, rendered_yaml: str) -> None:
    _, stderr, rc = await self._run_kubectl(
      "apply",
      "-f",
      "-",
      "-n",
      self._config.namespace,
      stdin=rendered_yaml,
    )
    if rc != 0:
      raise RuntimeError(f"Failed to apply Job: {stderr}")

  async def _poll_job(self, job_name: str) -> str:
    elapsed = 0
    while elapsed < self._config.timeout:
      stdout, _, rc = await self._run_kubectl(
        "get",
        "job",
        job_name,
        "-n",
        self._config.namespace,
        "-o",
        "jsonpath={.status.conditions[*].type}",
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
      "logs",
      f"job/{job_name}",
      "-c",
      "kernel-eval",
      "-n",
      self._config.namespace,
    )
    return stdout

  async def _cleanup(self, job_name: str) -> None:
    cm_name = f"{job_name}-payload"
    await self._run_kubectl(
      "delete",
      "job",
      job_name,
      "-n",
      self._config.namespace,
      "--ignore-not-found",
    )
    await self._run_kubectl(
      "delete",
      "configmap",
      cm_name,
      "-n",
      self._config.namespace,
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
