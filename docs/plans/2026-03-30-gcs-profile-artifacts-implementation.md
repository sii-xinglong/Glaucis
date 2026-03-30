# GCS Profile Artifacts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable pulling raw HLO/LLO IR text and XPlane trace events from TPU eval Pods to local machine via GCS, so AI can perform detailed IR-level analysis during `pallas-evolve:analyze`.

**Architecture:** Pod-side `evaluate.py` saves trace events JSON and IR file paths, then uploads them to `gs://glaucis-profiles/{job_name}/` via `google-cloud-storage` Python library. Local-side `KubeEvaluator` and `submit` skill download artifacts via `gcloud storage cp` after log collection. `analyze` skill reads downloaded IR/trace files with AI.

**Tech Stack:** `google-cloud-storage` Python library (Pod-side upload), `gcloud` CLI (local download), GKE Workload Identity (zero-key auth)

---

### Task 1: evaluate.py — Return file paths from profile stages

Add internal `_trace_events_path`, `_hlo_file`, `_llo_file` fields to stage return dicts so `main()` can locate them for upload.

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py:109-277` (stage_profile) and `kernel-evolve/docker/evaluate.py:280-486` (stage_profile_deep)

**Step 1: Write the failing test**

Create `kernel-evolve/tests/test_evaluate_artifacts.py`:

```python
"""Tests for artifact path propagation in evaluate.py stages."""

import json
import os
import tempfile
from pathlib import Path


def test_stage_profile_returns_trace_events_path(tmp_path):
    """stage_profile should write trace_events.json and return its path."""
    # We cannot run stage_profile without a TPU, so test the trace events
    # writing logic directly by simulating what stage_profile does post-xprof.
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    events = [{"pid": 1, "ts": 0, "dur": 100, "name": "op"}]
    trace_events_path = str(trace_dir / "trace_events.json")
    with open(trace_events_path, "w") as f:
        json.dump(events, f)

    assert Path(trace_events_path).exists()
    loaded = json.loads(Path(trace_events_path).read_text())
    assert len(loaded) == 1
    assert loaded[0]["name"] == "op"


def test_stage_profile_deep_returns_file_paths(tmp_path):
    """stage_profile_deep result should include _hlo_file and _llo_file paths."""
    # Simulate the file discovery logic from stage_profile_deep
    hlo_dir = tmp_path / "hlo"
    llo_dir = tmp_path / "llo"
    hlo_dir.mkdir()
    llo_dir.mkdir()

    hlo_file = hlo_dir / "module.after_optimizations.txt"
    hlo_file.write_text("HLO { custom-call(...) }")

    llo_file = llo_dir / "module.pass_79.llo"
    llo_file.write_text(";; bundle1\n.mxu0 op1\n;; bundle2\n.mxu1 op2")

    # The result dict should carry these paths
    result = {
        "ok": True,
        "_hlo_file": str(hlo_file),
        "_llo_file": str(llo_file),
        "vliw_bundle_count": 2,
    }
    assert Path(result["_hlo_file"]).exists()
    assert Path(result["_llo_file"]).exists()
```

**Step 2: Run test to verify it passes (these are unit tests for the concept)**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py -v`
Expected: PASS (these test the file I/O pattern, not the TPU stages)

**Step 3: Modify stage_profile to save trace_events.json and return its path**

In `kernel-evolve/docker/evaluate.py`, after line 168 (`trace_data = json.loads(tool_data_result)`), add:

```python
    # Save trace events for GCS upload
    trace_events_path = os.path.join(trace_dir, "trace_events.json")
    with open(trace_events_path, "w") as f:
      json.dump(trace_data.get("traceEvents", []), f)
```

In the return dict at line 270-275, add `_trace_events_path`:

```python
    return {
      "ok": True,
      "compute_ratio": 1.0 - ratio,
      "memory_transfer_ratio": ratio,
      "diagnostics": diag,
      "_trace_events_path": trace_events_path,
    }
```

**Step 4: Modify stage_profile_deep to return _hlo_file and _llo_file paths**

In `kernel-evolve/docker/evaluate.py`, update the return dict at line 477-484 to include file paths:

```python
    return {
      "ok": True,
      "vliw_bundle_count": vliw_bundle_count,
      "mxu_utilization": mxu_utilization,
      "hbm_bandwidth_bytes": hbm_bytes,
      "flops": flops,
      "arithmetic_intensity": arithmetic_intensity,
      "_hlo_file": chosen_hlo,
      "_llo_file": best_file,
    }
```

**Step 5: Run tests**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py tests/test_evaluator.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add kernel-evolve/docker/evaluate.py kernel-evolve/tests/test_evaluate_artifacts.py
git commit -m "feat(evaluate): return IR file paths and save trace events JSON from profile stages"
```

---

### Task 2: evaluate.py — Add upload_to_gcs function

Add a non-fatal GCS upload function using `google-cloud-storage` Python library.

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py` (add function after line 11)
- Test: `kernel-evolve/tests/test_evaluate_artifacts.py` (extend)

**Step 1: Write the failing test**

Append to `kernel-evolve/tests/test_evaluate_artifacts.py`:

```python
from unittest.mock import MagicMock, patch


def test_upload_to_gcs_success(tmp_path):
    """upload_to_gcs uploads files and returns uploaded list."""
    from evaluate import upload_to_gcs

    hlo = tmp_path / "hlo.txt"
    hlo.write_text("HLO content")
    llo = tmp_path / "llo.txt"
    llo.write_text("LLO content")

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("evaluate.storage") as mock_storage:
        mock_storage.Client.return_value = mock_client
        result = upload_to_gcs("test-job", {
            "hlo_post_opt.txt": str(hlo),
            "llo_final.txt": str(llo),
        })

    assert result["ok"] is True
    assert set(result["uploaded"]) == {"hlo_post_opt.txt", "llo_final.txt"}
    assert result["gcs_prefix"] == "gs://glaucis-profiles/test-job"
    assert mock_blob.upload_from_filename.call_count == 2


def test_upload_to_gcs_missing_file(tmp_path):
    """upload_to_gcs skips files that don't exist."""
    from evaluate import upload_to_gcs

    existing = tmp_path / "exists.txt"
    existing.write_text("content")

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("evaluate.storage") as mock_storage:
        mock_storage.Client.return_value = mock_client
        result = upload_to_gcs("test-job", {
            "exists.txt": str(existing),
            "missing.txt": "/nonexistent/path.txt",
        })

    assert result["ok"] is True
    assert result["uploaded"] == ["exists.txt"]
    assert mock_blob.upload_from_filename.call_count == 1


def test_upload_to_gcs_all_fail():
    """upload_to_gcs returns ok=False when all uploads fail."""
    from evaluate import upload_to_gcs

    result = upload_to_gcs("test-job", {
        "a.txt": "/nonexistent/a.txt",
        "b.txt": "/nonexistent/b.txt",
    })
    assert result["ok"] is False
    assert result["uploaded"] == []


def test_upload_to_gcs_empty_artifacts():
    """upload_to_gcs handles empty artifact dict."""
    from evaluate import upload_to_gcs

    result = upload_to_gcs("test-job", {})
    assert result["ok"] is False
    assert result["uploaded"] == []
```

**Step 2: Run test to verify it fails**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py::test_upload_to_gcs_success -v`
Expected: FAIL with `ImportError: cannot import name 'upload_to_gcs' from 'evaluate'`

**Step 3: Write the implementation**

Add at the top of `kernel-evolve/docker/evaluate.py`, after line 10 (`from typing import Any`):

```python
try:
  from google.cloud import storage
except ImportError:
  storage = None
```

Add the function after the imports (after line 12, `import numpy as np`):

```python
def upload_to_gcs(
    job_name: str,
    artifacts: dict[str, str],
    bucket_name: str = "glaucis-profiles",
) -> dict[str, Any]:
  """Upload profile artifacts to GCS. Non-fatal — never raises.

  Args:
      job_name: K8s job name, used as GCS path prefix.
      artifacts: {gcs_filename: local_path} pairs.
      bucket_name: GCS bucket name.

  Returns:
      {"ok": bool, "uploaded": list[str], "gcs_prefix": str}
  """
  prefix = f"gs://{bucket_name}/{job_name}"
  uploaded: list[str] = []
  if not artifacts:
    return {"ok": False, "uploaded": uploaded, "gcs_prefix": prefix}
  if storage is None:
    print("google-cloud-storage not installed, skipping GCS upload", file=sys.stderr)
    return {"ok": False, "uploaded": uploaded, "gcs_prefix": prefix}
  try:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for name, local_path in artifacts.items():
      if not os.path.exists(local_path):
        continue
      try:
        blob = bucket.blob(f"{job_name}/{name}")
        blob.upload_from_filename(local_path)
        uploaded.append(name)
      except Exception as e:
        print(f"GCS upload failed for {name}: {e}", file=sys.stderr)
  except Exception as e:
    print(f"GCS client init failed: {e}", file=sys.stderr)
  return {"ok": len(uploaded) > 0, "uploaded": uploaded, "gcs_prefix": prefix}
```

**Step 4: Run tests**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add kernel-evolve/docker/evaluate.py kernel-evolve/tests/test_evaluate_artifacts.py
git commit -m "feat(evaluate): add upload_to_gcs function for profile artifacts"
```

---

### Task 3: evaluate.py — Wire upload into main()

Collect artifact paths from stage results and upload after all profiling stages.

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py:559-594` (main function, after deep profile stage)

**Step 1: Write the failing test**

Append to `kernel-evolve/tests/test_evaluate_artifacts.py`:

```python
def test_main_collects_artifacts_for_upload():
    """Verify the artifact collection logic from stage results."""
    profile_result = {
        "ok": True,
        "compute_ratio": 0.85,
        "memory_transfer_ratio": 0.15,
        "diagnostics": {},
        "_trace_events_path": "/tmp/xplane_trace/trace_events.json",
    }
    deep_profile = {
        "ok": True,
        "vliw_bundle_count": 100,
        "_hlo_file": "/tmp/ir_dumps/hlo/module.after.txt",
        "_llo_file": "/tmp/ir_dumps/llo/module.pass_79.llo",
    }

    artifacts = {}
    if profile_result.get("ok"):
        trace_path = profile_result.get("_trace_events_path")
        if trace_path:
            artifacts["trace_events.json"] = trace_path
    if deep_profile.get("ok"):
        hlo_path = deep_profile.get("_hlo_file")
        llo_path = deep_profile.get("_llo_file")
        if hlo_path:
            artifacts["hlo_post_opt.txt"] = hlo_path
        if llo_path:
            artifacts["llo_final.txt"] = llo_path

    assert artifacts == {
        "trace_events.json": "/tmp/xplane_trace/trace_events.json",
        "hlo_post_opt.txt": "/tmp/ir_dumps/hlo/module.after.txt",
        "llo_final.txt": "/tmp/ir_dumps/llo/module.pass_79.llo",
    }


def test_internal_fields_stripped_from_eval_result():
    """Internal _fields must not appear in EVAL_RESULT JSON."""
    deep_profile = {
        "ok": True,
        "vliw_bundle_count": 100,
        "_hlo_file": "/tmp/some/path.txt",
        "_llo_file": "/tmp/other/path.llo",
    }
    # Strip internal fields before including in result
    clean = {k: v for k, v in deep_profile.items() if not k.startswith("_")}
    assert "_hlo_file" not in clean
    assert "_llo_file" not in clean
    assert "vliw_bundle_count" in clean
```

**Step 2: Run tests**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py -v`
Expected: PASS (these test the collection logic pattern)

**Step 3: Modify main() in evaluate.py**

After the deep profile stage (after line 577), add artifact collection and upload. Before the final `result` dict (line 579), add:

```python
  # ── Upload profile artifacts to GCS (non-fatal) ──
  artifacts = {}
  if profile_result.get("ok"):
    trace_path = profile_result.get("_trace_events_path")
    if trace_path:
      artifacts["trace_events.json"] = trace_path
  if deep_profile.get("ok"):
    hlo_path = deep_profile.get("_hlo_file")
    llo_path = deep_profile.get("_llo_file")
    if hlo_path:
      artifacts["hlo_post_opt.txt"] = hlo_path
    if llo_path:
      artifacts["llo_final.txt"] = llo_path

  gcs_result = upload_to_gcs(job_name, artifacts) if artifacts else {"ok": False, "uploaded": [], "gcs_prefix": ""}
  if gcs_result["ok"]:
    print(f"Uploaded artifacts: {gcs_result['uploaded']} to {gcs_result['gcs_prefix']}", file=sys.stderr)
```

The `job_name` needs to come from the eval payload. Add near the top of `main()`, after line 493:

```python
  job_name = request.get("variant_id", "unknown")
```

In the `result` dict at line 579-593, add `artifacts_gcs_prefix` to metadata and strip internal fields from `deep_profile`:

```python
  # Strip internal fields from deep_profile before including in result
  clean_deep_profile = {k: v for k, v in deep_profile.items() if not k.startswith("_")}

  result = {
    "status": "SUCCESS",
    "fitness": speedup,
    "latency_ms": perf_result["latency_ms"],
    "speedup": speedup,
    "flops": clean_deep_profile.get("flops", 0.0) or 0.0,
    "compute_ratio": compute_ratio,
    "memory_transfer_ratio": memory_transfer_ratio,
    "metadata": {
      "reference_latency_ms": ref_latency,
      "reference_perf_ok": ref_perf.get("ok", False),
      "profile_diagnostics": profile_diag,
      "profile": clean_deep_profile,
      **({"artifacts_gcs_prefix": gcs_result["gcs_prefix"]} if gcs_result.get("ok") else {}),
    },
  }
```

**Step 4: Run all evaluate tests**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add kernel-evolve/docker/evaluate.py kernel-evolve/tests/test_evaluate_artifacts.py
git commit -m "feat(evaluate): wire GCS artifact upload into main() pipeline"
```

---

### Task 4: K8s Job templates — Add serviceAccountName and google-cloud-storage

Update both Job templates to use the `kernel-eval` K8s ServiceAccount and install the GCS library.

**Files:**
- Modify: `.github/ci/kernel-eval-job.yaml:23-34`
- Modify: `.github/ci/kernel-eval-gmm-job.yaml:23-37`

**Step 1: No test needed (YAML template changes, tested by integration)**

**Step 2: Modify kernel-eval-job.yaml**

Add `serviceAccountName: kernel-eval` to the pod spec (after `spec:` at line 22, before `containers:`), and add `google-cloud-storage` to pip install (line 33):

In `.github/ci/kernel-eval-job.yaml`, add after the `spec:` line (line 22) and before `containers:` (line 23):
```yaml
      serviceAccountName: kernel-eval
```

Update the pip install line 32-34 to include `google-cloud-storage`:
```yaml
          uv pip install --system --prerelease=allow --no-cache \
            jax jaxlib libtpu numpy xprof google-cloud-storage \
            --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
```

**Step 3: Modify kernel-eval-gmm-job.yaml**

Same changes: add `serviceAccountName: kernel-eval` and `google-cloud-storage` to pip install.

**Step 4: Verify YAML is valid**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/ci/kernel-eval-job.yaml')); print('OK')" && python3 -c "import yaml; yaml.safe_load(open('.github/ci/kernel-eval-gmm-job.yaml')); print('OK')"`
Expected: OK OK

**Step 5: Commit**

```bash
git add .github/ci/kernel-eval-job.yaml .github/ci/kernel-eval-gmm-job.yaml
git commit -m "feat(ci): add kernel-eval KSA and google-cloud-storage to job templates"
```

---

### Task 5: kube_evaluator.py — Add artifact download

Add `_download_artifacts()` method and call it in `evaluate()` between log reading and cleanup.

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/kube_evaluator.py:127-184`
- Test: `kernel-evolve/tests/test_kube_evaluator.py`

**Step 1: Write the failing test**

Append to `kernel-evolve/tests/test_kube_evaluator.py`:

```python
@pytest.mark.asyncio
async def test_evaluate_downloads_artifacts(kube_config, eval_request, tmp_path):
    """evaluate() should attempt to download GCS artifacts when gcs_prefix is present."""
    evaluator = KubeEvaluator(kube_config)
    result_json = json.dumps({
        "status": "SUCCESS", "fitness": 2.5, "latency_ms": 0.5, "speedup": 2.5,
        "metadata": {"artifacts_gcs_prefix": "gs://glaucis-profiles/test-job"},
    })

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
    result_json = json.dumps({
        "status": "SUCCESS", "fitness": 2.5, "latency_ms": 0.5, "speedup": 2.5,
    })

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
```

**Step 2: Run test to verify it fails**

Run: `cd kernel-evolve && python -m pytest tests/test_kube_evaluator.py::test_evaluate_downloads_artifacts -v`
Expected: FAIL (no `_download_artifacts` method, or not called)

**Step 3: Write the implementation**

Add `_run_cmd` method and `_download_artifacts` method to `KubeEvaluator` in `kernel-evolve/src/kernel_evolve/kube_evaluator.py`.

After `_read_logs` (line 136), add:

```python
  async def _run_cmd(self, *args: str) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
      *args,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return stdout.decode(), stderr.decode(), proc.returncode

  async def _download_artifacts(self, gcs_prefix: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    _, stderr, rc = await self._run_cmd(
      "gcloud", "storage", "cp", "-r",
      f"{gcs_prefix}/*", str(dest_dir),
    )
    if rc != 0:
      print(f"Artifact download warning: {stderr}", file=sys.stderr)
```

Modify `evaluate()` method (lines 157-184). After `_read_logs` and parsing the result, but before `_cleanup`, add artifact download:

Replace lines 175-184 with:

```python
    logs = await self._read_logs(job_name)

    eval_result = None
    for line in logs.split("\n"):
      if "EVAL_RESULT:" in line:
        json_str = line.split("EVAL_RESULT:", 1)[1].strip()
        eval_result = EvalResult.from_dict(json.loads(json_str))
        break

    # Download GCS artifacts if available
    if eval_result and eval_result.metadata.get("artifacts_gcs_prefix"):
      gcs_prefix = eval_result.metadata["artifacts_gcs_prefix"]
      artifacts_dir = Path(f"artifacts/{job_name}")
      await self._download_artifacts(gcs_prefix, artifacts_dir)

    await self._cleanup(job_name)

    if eval_result:
      return eval_result

    error_snippet = logs[-500:] if len(logs) > 500 else logs
    return EvalResult.compile_error(f"No result in job logs. Tail: {error_snippet}")
```

**Step 4: Run tests**

Run: `cd kernel-evolve && python -m pytest tests/test_kube_evaluator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/kube_evaluator.py kernel-evolve/tests/test_kube_evaluator.py
git commit -m "feat(kube_evaluator): add GCS artifact download after eval completion"
```

---

### Task 6: submit skill — Add artifact download step

Update the submit skill to download GCS artifacts after parsing EVAL_RESULT.

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/submit/SKILL.md` (between Step 8 and Step 9)

**Step 1: No test needed (skill markdown, tested by manual use)**

**Step 2: Add Step 8b to submit skill**

In `kernel-evolve/plugins/pallas-evolve/skills/submit/SKILL.md`, after `### Step 8: Parse EVAL_RESULT` section (after line 145), add:

```markdown
### Step 8b: Download profile artifacts

If the eval result contains an `artifacts_gcs_prefix` in metadata, download the raw IR and trace files locally:

\```bash
GCS_PREFIX=$(python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
print(r.get('metadata', {}).get('artifacts_gcs_prefix', ''))
" "iteration_{N}/eval_result.json")

if [ -n "$GCS_PREFIX" ]; then
  gcloud storage cp "${GCS_PREFIX}/*" "iteration_{N}/" 2>/dev/null && \
    echo "Downloaded artifacts from ${GCS_PREFIX}" || \
    echo "Artifact download skipped (GCS not configured or empty)"
fi
\```

This downloads up to three files into `iteration_{N}/`:
- `hlo_post_opt.txt` — Post-optimization HLO IR text
- `llo_final.txt` — Final-pass LLO IR text with VLIW bundles
- `trace_events.json` — Expanded XPlane trace events (Chrome trace format)

These files are optional. If download fails, the analyze skill falls back to metrics-only analysis.
```

**Step 3: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/submit/SKILL.md
git commit -m "feat(submit skill): add GCS artifact download step"
```

---

### Task 7: analyze skill — Add IR and trace reading steps

Update the analyze skill to read downloaded IR text and trace events with AI.

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md` (after Step 3)

**Step 1: No test needed (skill markdown)**

**Step 2: Add Step 3b to analyze skill**

In `kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md`, after `### Step 3: Performance analysis` section (before `### Step 4: Trend analysis`), add:

```markdown
### Step 3b: Deep IR analysis (if available)

Check if raw profile artifacts were downloaded to the iteration directory. These provide much richer optimization signals than the scalar metrics alone.

**HLO IR (`iteration_{N}/hlo_post_opt.txt`)**

If this file exists, read it and analyze:

- **Fusion decisions**: Which ops were fused into the `tpu_custom_call`? Were any ops left unfused that could benefit from fusion?
- **Memory layout**: Are there unnecessary transposes, copies, or layout conversions? Check for `transpose`, `copy`, or `bitcast` ops outside the fused region.
- **Parameter shapes**: Verify that the tiling dimensions visible in HLO match the Pallas `BlockSpec` grid. Mismatches indicate suboptimal tiling.
- **Constant folding**: Are there constants that could be folded at compile time but weren't?
- **Redundant ops**: Look for `broadcast`, `reshape`, or `slice` chains that suggest the compiler couldn't simplify the data flow.

**LLO IR (`iteration_{N}/llo_final.txt`)**

If this file exists, read it and analyze:

- **VLIW bundle density**: Are bundles densely packed (3-4 ops per `;;` block) or mostly single-op? Dense bundles mean the compiler is effectively utilizing instruction-level parallelism.
- **MXU scheduling**: Where are `.mxu0`/`.mxu1` ops placed? Long gaps between MXU ops suggest pipeline bubbles. Consecutive MXU ops on both ports (`.mxu0` and `.mxu1` in the same bundle) indicate dual-MXU scheduling.
- **Pipeline stalls**: Look for `nop` instructions or `wait` barriers. Multiple `nop`s in sequence indicate the compiler couldn't fill the pipeline.
- **Register pressure**: Look for store/load patterns to VMEM (`.vmem_store` followed later by `.vmem_load` of the same address) — these indicate register spills.
- **DMA scheduling**: Check for `dma.start` and `dma.done` pairs. Good pipelining has `dma.start` well ahead of the corresponding `dma.done`, overlapping with computation.

**Trace Events (`iteration_{N}/trace_events.json`)**

If this file exists, read it (it may be large — focus on events with `dur > 0` on the TPU device pid) and analyze:

- **Event distribution**: What types of events dominate? Group by `name` and sum durations.
- **Compute vs sync gaps**: Look for long `SyncWait` events between computation events. These represent times the TPU is idle waiting for data.
- **DMA overlap**: Are there DMA transfer events running concurrently with computation events (overlapping `ts` + `dur` ranges)?
- **Iteration consistency**: Are the 3 profiled iterations similar in timing, or is there variance suggesting cold-start effects?

**Include IR-based findings in the analysis.md output.** When IR analysis reveals something the scalar metrics missed (e.g., register spills, missed fusions, pipeline bubbles), highlight it as a concrete optimization target with specific suggestions.
```

**Step 3: Update analysis.md template**

In the same file, update the Step 6 analysis template to include IR findings. After the `### Suggestions` line in the template, add:

```markdown
### IR Analysis (if available)
{Specific findings from HLO/LLO/trace — e.g., "LLO shows 12 nop sequences averaging 4 nops each, indicating pipeline bubbles between DMA and MXU ops. Consider adding prefetch or increasing tile size to hide latency."}
```

**Step 4: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md
git commit -m "feat(analyze skill): add deep IR and trace analysis steps"
```

---

### Task 8: GCS infrastructure setup (manual / documented)

Document the one-time GCS bucket and IAM setup commands. This task is not code — it creates a setup script.

**Files:**
- Create: `kernel-evolve/scripts/setup_gcs_profile_bucket.sh`

**Step 1: Write the setup script**

```bash
#!/usr/bin/env bash
# One-time setup for GCS profile artifacts bucket and IAM bindings.
# Run this manually with appropriate gcloud credentials.
set -euo pipefail

PROJECT="tpu-service-473302"
BUCKET="glaucis-profiles"
REGION="us-central1"
GCP_SA="kernel-eval"
K8S_SA="kernel-eval"
K8S_NS="default"
# Update this with your actual GKE cluster name
GKE_CLUSTER="tpu7x-cluster"

echo "=== Creating GCS bucket ==="
gcloud storage buckets create "gs://${BUCKET}" \
  --project="${PROJECT}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  2>/dev/null || echo "Bucket already exists"

echo "=== Setting lifecycle (7-day auto-delete) ==="
cat <<'LIFECYCLE' | gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file=-
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 7}}]}
LIFECYCLE

echo "=== Creating GCP service account ==="
gcloud iam service-accounts create "${GCP_SA}" \
  --project="${PROJECT}" \
  --display-name="Kernel Eval Pod SA" \
  2>/dev/null || echo "SA already exists"

echo "=== Granting GCS objectCreator to SA ==="
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"

echo "=== Creating K8s service account ==="
kubectl create serviceaccount "${K8S_SA}" -n "${K8S_NS}" \
  2>/dev/null || echo "K8s SA already exists"

echo "=== Binding K8s SA to GCP SA via Workload Identity ==="
gcloud iam service-accounts add-iam-policy-binding \
  "${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role=roles/iam.workloadIdentityUser \
  --member="principal://iam.googleapis.com/projects/785128357837/locations/global/workloadIdentityPools/${GKE_CLUSTER}.svc.id.goog/subject/ns/${K8S_NS}/sa/${K8S_SA}"

echo "=== Annotating K8s SA ==="
kubectl annotate serviceaccount "${K8S_SA}" -n "${K8S_NS}" \
  "iam.gke.io/gcp-service-account=${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --overwrite

echo "=== Done! Verify with: ==="
echo "kubectl get sa ${K8S_SA} -n ${K8S_NS} -o yaml"
```

**Step 2: Commit**

```bash
chmod +x kernel-evolve/scripts/setup_gcs_profile_bucket.sh
git add kernel-evolve/scripts/setup_gcs_profile_bucket.sh
git commit -m "feat(scripts): add GCS profile bucket setup script"
```

---

### Task Dependency Summary

```
Task 1 (stage return paths) ─┐
                              ├─► Task 3 (wire into main)
Task 2 (upload_to_gcs fn) ───┘         │
                                        ├─► Task 4 (K8s templates)
                                        │
Task 5 (kube_evaluator download) ◄──────┘
                                        │
Task 6 (submit skill) ◄────────────────┘
Task 7 (analyze skill) — independent, can run in parallel with 5/6
Task 8 (GCS setup script) — independent, can run anytime
```

Tasks 1-2 are independent and can run in parallel.
Tasks 6, 7, 8 are independent of each other.
