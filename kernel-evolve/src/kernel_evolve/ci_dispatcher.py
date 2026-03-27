"""CI-based evaluation dispatcher using GitHub Actions workflows."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from kernel_evolve.evaluator import EvalRequest, EvalResult, Evaluator


@dataclass
class CIConfig:
  repo: str
  workflow: str
  branch: str = "main"
  poll_interval: int = 30
  timeout: int = 600


class CIDispatcher(Evaluator):
  """Dispatches kernel evaluation via GitHub Actions CI workflows."""

  def __init__(self, config: CIConfig):
    self._config = config

  async def evaluate(self, request: EvalRequest) -> EvalResult:
    try:
      run_id = await self._trigger_workflow(request)
    except Exception as e:
      return EvalResult.compile_error(f"Failed to trigger CI: {e}")

    status = await self._wait_for_completion(run_id)

    if status != "completed":
      return EvalResult.compile_error(f"CI run {status}: timed out or failed")

    return await self._collect_result(run_id)

  async def _trigger_workflow(self, request: EvalRequest) -> str:
    payload = request.encode_b64()
    cmd = [
      "gh",
      "workflow",
      "run",
      self._config.workflow,
      "--repo",
      self._config.repo,
      "--ref",
      self._config.branch,
      "-f",
      f"ref={self._config.branch}",
      "-f",
      f"eval_payload={payload}",
      "-f",
      f"variant_id={request.variant_id}",
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
      raise RuntimeError(f"gh workflow run failed: {stderr.decode()}")

    list_cmd = [
      "gh",
      "run",
      "list",
      "--repo",
      self._config.repo,
      "--workflow",
      self._config.workflow,
      "--limit",
      "1",
      "--json",
      "databaseId",
    ]
    proc = await asyncio.create_subprocess_exec(
      *list_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    runs = json.loads(stdout.decode())
    return str(runs[0]["databaseId"])

  async def _wait_for_completion(self, run_id: str) -> str:
    elapsed = 0
    while elapsed < self._config.timeout:
      cmd = [
        "gh",
        "run",
        "view",
        run_id,
        "--repo",
        self._config.repo,
        "--json",
        "status,conclusion",
      ]
      proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
      stdout, _ = await proc.communicate()
      data = json.loads(stdout.decode())
      if data["status"] == "completed":
        return data.get("conclusion", "completed")
      await asyncio.sleep(self._config.poll_interval)
      elapsed += self._config.poll_interval
    return "timed_out"

  async def _collect_result(self, run_id: str) -> EvalResult:
    cmd = [
      "gh",
      "run",
      "view",
      run_id,
      "--repo",
      self._config.repo,
      "--log",
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, _ = await proc.communicate()
    log_text = stdout.decode()

    for line in log_text.split("\n"):
      if "EVAL_RESULT:" in line:
        json_str = line.split("EVAL_RESULT:", 1)[1].strip()
        return EvalResult.from_dict(json.loads(json_str))

    return EvalResult.compile_error("Could not find EVAL_RESULT in CI logs")
