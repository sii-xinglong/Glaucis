---
name: submit
description: Use when submitting a batch of Pallas kernel variants for TPU evaluation via kubectl — creates single K8s Job that evaluates N variants serially, collects all EVAL_RESULTs
---

# Submit Batch of Kernel Variants for TPU Evaluation

Submit a batch of kernel variants for evaluation on GKE TPU v7x via a single K8s Job. Builds a combined payload with all variants, creates a ConfigMap, deploys the Job, and collects per-variant results.

## Context

This skill is invoked by `pallas-evolve:start` during the optimization loop, or standalone for debugging. It expects:
- A run directory with the current iteration's `variants/` subdirectory containing one or more variant kernels
- A config YAML already loaded in context (kernel paths, shapes, evaluator settings)

## Procedure

### Step 1: Locate files

Find the current iteration directory (the latest `iteration_{N}/` in the run dir). Identify:
- ALL variant kernel files: `iteration_{N}/variants/*/kernel.py`
- The reference kernel from the config's `kernel.reference` path (relative to config dir)
- The config's `shapes`, `correctness.rtol`, `correctness.atol`

Count the number of variants (`N_VARIANTS`). If zero, stop with an error — nothing to submit.

### Step 2: Construct batch payload

Run this Python script via Bash to build the base64-encoded batch payload:

```bash
python3 -c "
import json, base64, sys, glob, os

reference_code = open(sys.argv[1]).read()
shapes = json.loads(sys.argv[2])
rtol = float(sys.argv[3])
atol = float(sys.argv[4])
variant_dir = sys.argv[5]

variants = []
for vdir in sorted(glob.glob(os.path.join(variant_dir, '*/kernel.py'))):
    variant_id = os.path.basename(os.path.dirname(vdir))
    kernel_code = open(vdir).read()
    variants.append({'variant_id': variant_id, 'kernel_code': kernel_code})

payload = json.dumps({
    'batch': True,
    'reference_code': reference_code,
    'shapes': shapes,
    'rtol': rtol,
    'atol': atol,
    'variants': variants
})

print(base64.b64encode(payload.encode()).decode())
" \
  "<path/to/reference_kernel.py>" \
  '<shapes_json_array>' \
  "{rtol}" \
  "{atol}" \
  "<path/to/iteration_N/variants>"
```

Save the output as `B64_PAYLOAD`.

### Step 3: Payload size check

Check the size of the base64-encoded payload:

```bash
echo -n "${B64_PAYLOAD}" | wc -c
```

If the payload exceeds 900KB (921600 bytes):
1. Decode back to raw JSON and save to a temp file
2. Upload to GCS:
   ```bash
   echo "${B64_PAYLOAD}" | base64 -d > /tmp/payload.json
   gcloud storage cp /tmp/payload.json "gs://glaucis-profiles/${JOB_NAME}/payload.json"
   rm /tmp/payload.json
   ```
3. Replace `B64_PAYLOAD` with the base64 encoding of a pointer JSON:
   ```bash
   B64_PAYLOAD=$(echo -n '{"gcs_payload":"gs://glaucis-profiles/'${JOB_NAME}'/payload.json"}' | base64)
   ```

> **Note:** The pod-side evaluator must detect `gcs_payload` and download the full payload. If this is not yet implemented, warn the user that payloads above 900KB may not work and consider reducing variant count.

### Step 4: Generate job name

Include the kernel name from the config and a timestamp to prevent conflicts between concurrent optimizations:

```bash
KERNEL_NAME_SLUG=$(echo "{kernel_name}" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]-' '-' | sed 's/--*/-/g; s/^-//; s/-$//')
TIMESTAMP=$(date +%m%d-%H%M%S)
JOB_NAME="${KERNEL_NAME_SLUG}-iter${N}-${TIMESTAMP}"
```

Job names must be: lowercase, alphanumeric + hyphens, max 63 chars. Truncate if needed:

```bash
JOB_NAME=$(echo "${JOB_NAME}" | cut -c1-63 | sed 's/-$//')
```

Example: optimizing `chunk_gla` at iteration 3 → `chunk-gla-iter3-0330-142517`.

### Step 5: Create ConfigMap

```bash
kubectl create configmap ${JOB_NAME}-payload \
  --from-literal=payload="${B64_PAYLOAD}" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl label configmap ${JOB_NAME}-payload app=kernel-eval --overwrite
```

### Step 6: Deploy K8s Job

Read the job template path from the config's `evaluator.job_template` (relative to repo root). Render with variable substitution and apply:

```bash
ACTIVE_DEADLINE=$((N_VARIANTS * 300 + 300))

python3 -c "
import string, sys
tmpl = open(sys.argv[1]).read()
rendered = string.Template(tmpl).safe_substitute(
    JOB_NAME=sys.argv[2],
    BRANCH=sys.argv[3],
    REPO=sys.argv[4],
    VARIANT_ID=sys.argv[5],
    ACTIVE_DEADLINE=sys.argv[6]
)
print(rendered)
" \
  "{job_template_path}" \
  "${JOB_NAME}" \
  "{branch}" \
  "{repo}" \
  "batch-${N_VARIANTS}" \
  "${ACTIVE_DEADLINE}" | kubectl apply -f -
```

### Step 7: Wait for completion

Calculate the timeout based on variant count:

```bash
TIMEOUT=$((N_VARIANTS * 300 + 300))
kubectl wait --for=condition=complete --timeout=${TIMEOUT}s job/${JOB_NAME} 2>/dev/null && echo "JOB_STATUS:Complete" || \
kubectl wait --for=condition=failed --timeout=5s job/${JOB_NAME} 2>/dev/null && echo "JOB_STATUS:Failed" || \
echo "JOB_STATUS:Timeout"
```

Use a Bash tool call with `timeout: {TIMEOUT * 1000 + 30000}` (timeout in ms, plus buffer).

On timeout or failure, check the job status for diagnostics:
```bash
kubectl get job/${JOB_NAME} -o jsonpath='{.status.conditions}'
```

Even on failure/timeout, proceed to Step 8 to collect any partial results.

### Step 8: Collect and parse results

Fetch logs from the job:

```bash
kubectl logs job/${JOB_NAME} -c kernel-eval
```

Scan the logs for ALL lines containing `EVAL_RESULT:`. Each line is one variant's result JSON. Parse each result and match `variant_id` to the directory name under `iteration_{N}/variants/`.

Use a Python script to parse and distribute results:

```bash
python3 -c "
import json, sys, os

logs = sys.stdin.read()
variants_dir = sys.argv[1]

# Collect all EVAL_RESULT lines
results = {}
for line in logs.splitlines():
    if 'EVAL_RESULT:' in line:
        json_str = line.split('EVAL_RESULT:', 1)[1].strip()
        try:
            result = json.loads(json_str)
            vid = result.get('variant_id', '')
            results[vid] = result
        except json.JSONDecodeError:
            pass

# Get all variant directories
variant_dirs = [d for d in os.listdir(variants_dir)
                if os.path.isdir(os.path.join(variants_dir, d))]

for vdir in variant_dirs:
    out_path = os.path.join(variants_dir, vdir, 'eval_result.json')
    if vdir in results:
        with open(out_path, 'w') as f:
            json.dump(results[vdir], f, indent=2)
        print(f'Saved result for {vdir}: status={results[vdir].get(\"status\", \"UNKNOWN\")}')
    else:
        # Synthetic error for missing results
        synthetic = {
            'variant_id': vdir,
            'status': 'COMPILE_ERROR',
            'fitness': 0.0,
            'error': f'No EVAL_RESULT found for variant {vdir} in job logs',
            'latency_ms': 0.0,
            'speedup': 0.0
        }
        with open(out_path, 'w') as f:
            json.dump(synthetic, f, indent=2)
        print(f'Saved synthetic error for {vdir}: no result in logs')
" "<path/to/iteration_N/variants>" <<< "$(kubectl logs job/${JOB_NAME} -c kernel-eval)"
```

### Step 8b: Download artifacts

For each variant that has a result with `metadata.artifacts_gcs_prefix`, download IR and trace files:

```bash
for VARIANT_DIR in iteration_N/variants/*/; do
  VARIANT_NAME=$(basename "$VARIANT_DIR")
  RESULT_FILE="${VARIANT_DIR}eval_result.json"

  if [ -f "$RESULT_FILE" ]; then
    GCS_PREFIX=$(python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
print(r.get('metadata', {}).get('artifacts_gcs_prefix', ''))
" "$RESULT_FILE")

    if [ -n "$GCS_PREFIX" ]; then
      gcloud storage cp -r "${GCS_PREFIX}/*" "${VARIANT_DIR}" 2>/dev/null && \
        echo "Downloaded artifacts for ${VARIANT_NAME} from ${GCS_PREFIX}" || \
        echo "Artifact download skipped for ${VARIANT_NAME} (GCS not configured or empty)"
    fi
  fi
done
```

This downloads up to three files per variant:
- `hlo_post_opt.txt` — Post-optimization HLO IR text
- `llo_final.txt` — Final-pass LLO IR text with VLIW bundles
- `trace_events.json` — Expanded XPlane trace events (Chrome trace format)

These files are optional. If download fails, the analyze skill falls back to metrics-only analysis.

### Step 9: Cleanup

Always clean up, even on failure:

```bash
kubectl delete job ${JOB_NAME} --ignore-not-found
kubectl delete configmap ${JOB_NAME}-payload --ignore-not-found
```

## Error Handling

- If ConfigMap creation fails: save synthetic error results for ALL variants, skip to cleanup
- If Job apply fails: save synthetic error results for ALL variants, clean up ConfigMap
- If Job times out or fails: still collect partial results from logs (some variants may have succeeded), then clean up
- If log collection fails: save synthetic error results for all variants with "Failed to collect logs"
- If a variant has no matching `EVAL_RESULT:` in the logs: save a synthetic COMPILE_ERROR for that variant
- Partial results are acceptable — some variants may succeed while others fail
- Always run cleanup step (Step 9) regardless of outcome
