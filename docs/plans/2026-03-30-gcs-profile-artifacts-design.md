# GCS Profile Artifacts: Local AI Analysis of Raw IR and Trace Data

## Problem

Currently, all profile data is generated, parsed, and reduced to scalar metrics inside the TPU Pod. Only the `EVAL_RESULT:{json}` line (a few KB) is transmitted back via `kubectl logs`. The raw artifacts -- HLO text, LLO text, and XPlane trace events -- are destroyed when the Pod terminates.

This prevents the AI from:
- Reading HLO IR to understand compiler decisions (fusion, layout, memory allocation)
- Reading LLO IR to identify pipeline stalls, register pressure, VLIW scheduling issues
- Analyzing individual trace events to understand compute/sync patterns at microsecond granularity

## Solution

Use GCS as an intermediate store. The Pod uploads raw IR text files and an expanded trace event JSON after profiling completes. The local client downloads these artifacts before Pod cleanup. The AI reads them directly during the `pallas-evolve:analyze` phase.

## Architecture

```
Pod (evaluate.py)                         GCS                         Local
  stage_profile()                         gs://glaucis-profiles/
    └─ xplane.pb ─► xprof parse             /{job_name}/
       └─ trace_events.json ──────► upload ──► trace_events.json
  stage_profile_deep()                                                iteration_N/
    ├─ hlo_dir/ ─► find post-opt             /{job_name}/              ├─ eval_result.json
    │  └─ hlo_post_opt.txt ───────► upload ──► hlo_post_opt.txt ──► download ──► hlo_post_opt.txt
    ├─ llo_dir/ ─► find final pass                                     ├─ llo_final.txt
    │  └─ llo_final.txt ──────────► upload ──► llo_final.txt ──── ► download ──► llo_final.txt
    └─ EVAL_RESULT:{json} ────────► kubectl logs (unchanged) ──────► eval_result.json
                                                                       └─ trace_events.json
```

## Data Scope

| Artifact | Source | Typical Size | Content |
|----------|--------|-------------|---------|
| `hlo_post_opt.txt` | `stage_profile_deep()` HLO dump, file with "after" in name | 10-200 KB | Post-optimization HLO text IR |
| `llo_final.txt` | `stage_profile_deep()` LLO dump, highest pass number | 10-500 KB | Final-pass LLO with VLIW bundles, MXU annotations |
| `trace_events.json` | `stage_profile()` xprof `xspace_to_tool_data` output, `traceEvents` array | 50-500 KB | Chrome trace format events (pid, tid, ts, dur, name, args) |

Total per evaluation: ~100 KB - 1 MB. XPlane protobuf (1-50 MB) is NOT uploaded; the expanded JSON event list is sufficient for AI analysis.

## GCS Setup

### Bucket

- Name: `glaucis-profiles`
- Location: `us-central1` (same region as GKE cluster)
- Storage class: Standard
- Lifecycle: Delete objects after 7 days (profile artifacts are ephemeral)

```bash
gcloud storage buckets create gs://glaucis-profiles \
  --project=tpu-service-473302 \
  --location=us-central1 \
  --uniform-bucket-level-access \
  --lifecycle-file=- <<'EOF'
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 7}}]}
EOF
```

### IAM: Workload Identity Binding

The Pod needs GCS write access. Use GKE Workload Identity to bind a K8s ServiceAccount to a GCP SA.

1. Create GCP SA:
```bash
gcloud iam service-accounts create kernel-eval \
  --project=tpu-service-473302 \
  --display-name="Kernel Eval Pod SA"
```

2. Grant `roles/storage.objectCreator` on the bucket:
```bash
gcloud storage buckets add-iam-policy-binding gs://glaucis-profiles \
  --member="serviceAccount:kernel-eval@tpu-service-473302.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"
```

3. Create K8s ServiceAccount and bind via WI:
```bash
kubectl create serviceaccount kernel-eval -n default

gcloud iam service-accounts add-iam-policy-binding \
  kernel-eval@tpu-service-473302.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="principal://iam.googleapis.com/projects/785128357837/locations/global/workloadIdentityPools/tpu7x-cluster.svc.id.goog/subject/ns/default/sa/kernel-eval"
```

4. Annotate K8s SA:
```bash
kubectl annotate serviceaccount kernel-eval -n default \
  iam.gke.io/gcp-service-account=kernel-eval@tpu-service-473302.iam.gserviceaccount.com
```

## Code Changes

### 1. evaluate.py: Upload function

Add `upload_to_gcs()` at module level. Uses `subprocess` to call `gcloud storage cp` (available in GKE node images). Falls back gracefully -- upload failure does not break the evaluation pipeline.

```python
def upload_to_gcs(job_name: str, artifacts: dict[str, str]) -> dict:
    """Upload profile artifacts to GCS. Non-fatal.

    Args:
        job_name: K8s job name, used as GCS prefix.
        artifacts: {gcs_filename: local_path} pairs.

    Returns:
        {"ok": bool, "uploaded": list[str], "gcs_prefix": str}
    """
    import subprocess
    prefix = f"gs://glaucis-profiles/{job_name}"
    uploaded = []
    for name, local_path in artifacts.items():
        if not os.path.exists(local_path):
            continue
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", local_path, f"{prefix}/{name}"],
                check=True, capture_output=True, timeout=60,
            )
            uploaded.append(name)
        except Exception as e:
            print(f"GCS upload failed for {name}: {e}", file=sys.stderr)
    return {"ok": len(uploaded) > 0, "uploaded": uploaded, "gcs_prefix": prefix}
```

### 2. evaluate.py: Capture trace events JSON

In `stage_profile()`, after parsing XPlane with xprof, write the full `traceEvents` array to a temp file:

```python
# After line 168: trace_data = json.loads(tool_data_result)
trace_events_path = os.path.join(trace_dir, "trace_events.json")
with open(trace_events_path, "w") as f:
    json.dump(trace_data.get("traceEvents", []), f)
# Return the path in the result dict for upload
```

### 3. evaluate.py: main() orchestration

After all profiling stages, collect artifact paths and upload:

```python
# After deep_profile stage (around line 570)
artifacts = {}

# IR files from stage_profile_deep
if deep_profile.get("ok"):
    hlo_path = deep_profile.get("_hlo_file")  # new internal field
    llo_path = deep_profile.get("_llo_file")   # new internal field
    if hlo_path: artifacts["hlo_post_opt.txt"] = hlo_path
    if llo_path: artifacts["llo_final.txt"] = llo_path

# Trace events from stage_profile
trace_path = profile_result.get("_trace_events_path")  # new internal field
if trace_path: artifacts["trace_events.json"] = trace_path

# Upload (non-fatal)
gcs_result = upload_to_gcs(job_name, artifacts) if artifacts else {"ok": False}
if gcs_result.get("ok"):
    result["metadata"]["artifacts_gcs_prefix"] = gcs_result["gcs_prefix"]
```

Note: `stage_profile_deep()` will return `_hlo_file` and `_llo_file` as internal fields (the paths to the chosen HLO/LLO files it already finds). These fields are stripped before serialization to `EVAL_RESULT`.

### 4. kernel-eval-job.yaml: Add serviceAccountName and install gcloud

```yaml
spec:
  serviceAccountName: kernel-eval    # NEW
  containers:
  - name: kernel-eval
    command:
    - bash
    - -c
    - |
      set -ex
      pip install uv
      uv pip install --system --prerelease=allow --no-cache \
        jax jaxlib libtpu numpy xprof \
        --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
      # gcloud is already available via GKE node tooling / google-cloud-sdk
      ...
```

The `python:3.12` base image does NOT include `gcloud`. Two options:
- **Option A**: Switch to `google/cloud-sdk:slim` as base image (includes gcloud + Python)
- **Option B**: Use the `google-cloud-storage` Python library instead of `gcloud` CLI

Recommend **Option B**: add `google-cloud-storage` to pip install, use Python API in `upload_to_gcs()`. This avoids changing the base image.

```python
def upload_to_gcs(job_name: str, artifacts: dict[str, str]) -> dict:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("glaucis-profiles")
    uploaded = []
    for name, local_path in artifacts.items():
        if not os.path.exists(local_path):
            continue
        try:
            blob = bucket.blob(f"{job_name}/{name}")
            blob.upload_from_filename(local_path)
            uploaded.append(name)
        except Exception as e:
            print(f"GCS upload failed for {name}: {e}", file=sys.stderr)
    return {"ok": len(uploaded) > 0, "uploaded": uploaded,
            "gcs_prefix": f"gs://glaucis-profiles/{job_name}"}
```

### 5. kube_evaluator.py: Download artifacts

Add `_download_artifacts()` method, called after `_read_logs()` and before `_cleanup()`:

```python
async def _download_artifacts(self, job_name: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    gcs_prefix = f"gs://glaucis-profiles/{job_name}/"
    stdout, stderr, rc = await self._run_kubectl_host(
        "gcloud", "storage", "cp", "-r", gcs_prefix, str(dest_dir)
    )
    if rc != 0:
        print(f"Artifact download warning: {stderr}", file=sys.stderr)
```

Actually, since `_run_kubectl` is kubectl-specific, add a general `_run_cmd` or use `asyncio.create_subprocess_exec` directly.

### 6. submit skill: Download step

Between Step 8 (parse EVAL_RESULT) and Step 9 (cleanup), add:

```markdown
### Step 8b: Download profile artifacts

If the eval_result contains `metadata.artifacts_gcs_prefix`:

\```bash
GCS_PREFIX=$(python3 -c "import json; r=json.load(open('iteration_{N}/eval_result.json')); print(r.get('metadata',{}).get('artifacts_gcs_prefix',''))")
if [ -n "$GCS_PREFIX" ]; then
  gcloud storage cp -r "${GCS_PREFIX}/*" "iteration_{N}/"
fi
\```

This downloads `hlo_post_opt.txt`, `llo_final.txt`, and `trace_events.json` to the iteration directory.
```

### 7. analyze skill: Read IR and trace data

Add new analysis steps after Step 3:

```markdown
### Step 3b: Deep IR analysis (if available)

If `iteration_{N}/hlo_post_opt.txt` exists, read it and analyze:
- **Fusion decisions**: which ops were fused into the custom_call? Any missed fusions?
- **Memory layout**: check for unnecessary transposes or copies
- **Parameter shapes**: verify expected tiling shows up in the IR

If `iteration_{N}/llo_final.txt` exists, read it and analyze:
- **VLIW scheduling quality**: are bundles densely packed or mostly single-op?
- **MXU usage patterns**: where are .mxu0/.mxu1 ops placed? Any long gaps?
- **Pipeline stalls**: look for `nop` sequences or `wait` instructions
- **Register pressure**: check for spills to VMEM (store/load patterns around computation)

If `iteration_{N}/trace_events.json` exists, read it and analyze:
- **Event timeline**: which events dominate the trace? Any unexpected gaps?
- **DMA patterns**: are memory transfers overlapping with computation?
- **Kernel launch overhead**: time between events
```

## Backward Compatibility

- `EVAL_RESULT` JSON channel unchanged. All existing scalar metrics are still there.
- `artifacts_gcs_prefix` is an optional new field in `metadata`. Old clients ignore it.
- GCS upload failure is non-fatal: `upload_to_gcs()` catches all exceptions and returns `ok=False`.
- If GCS artifacts are not downloaded, the analyze skill falls back to its current behavior (metrics-only analysis).
- `_hlo_file` and `_llo_file` internal fields are stripped before `EVAL_RESULT` serialization.

## Security

- K8s SA `kernel-eval` has minimal permissions: `roles/storage.objectCreator` on a single bucket.
- No read-back from GCS inside the Pod (write-only).
- Bucket uses uniform bucket-level access (no per-object ACLs).
- 7-day lifecycle auto-deletes stale artifacts.
- No secrets or credentials stored in the repo; WI handles authentication.

## Testing

- Unit test `upload_to_gcs()` with mocked `google.cloud.storage.Client`
- Unit test `_download_artifacts()` with mocked subprocess
- Integration test: submit a small matmul kernel eval, verify GCS artifacts appear and can be downloaded
- Verify analyze skill handles missing artifacts gracefully (no GCS configured)
