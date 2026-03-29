---
name: submit
description: Use when submitting a Pallas kernel for TPU evaluation via kubectl — creates K8s Job, polls for completion, collects EVAL_RESULT
---

# Submit Kernel for TPU Evaluation

Submit a kernel variant for evaluation on GKE TPU v7x via a K8s Job. Creates a ConfigMap with the eval payload, deploys the Job, polls until completion, and collects the result.

## Context

This skill is invoked by `pallas-evolve:start` during the optimization loop, or standalone for debugging. It expects:
- A run directory with the current iteration's `kernel.py`
- A config YAML already loaded in context (kernel paths, shapes, evaluator settings)

## Procedure

### Step 1: Locate files

Find the current iteration directory (the latest `iteration_{N}/` in the run dir). Read:
- `iteration_{N}/kernel.py` — the kernel to evaluate
- The reference kernel from the config's `kernel.reference` path (relative to config dir)
- The config's `shapes`, `correctness.rtol`, `correctness.atol`

### Step 2: Construct eval payload

Run this Python script via Bash to create the base64-encoded payload:

```bash
python3 -c "
import json, base64, sys
kernel_code = open(sys.argv[1]).read()
reference_code = open(sys.argv[2]).read()
shapes = json.loads(sys.argv[3])
payload = json.dumps({
    'variant_id': sys.argv[4],
    'kernel_code': kernel_code,
    'reference_code': reference_code,
    'shapes': shapes,
    'rtol': float(sys.argv[5]),
    'atol': float(sys.argv[6])
})
print(base64.b64encode(payload.encode()).decode())
" \
  "<path/to/iteration_N/kernel.py>" \
  "<path/to/reference_kernel.py>" \
  '<shapes_json_array>' \
  "iter-{N}" \
  "{rtol}" \
  "{atol}"
```

Save the output as `EVAL_PAYLOAD`.

### Step 3: Generate job name

```bash
JOB_NAME="kernel-eval-iter-{N}"
```

Job names must be: lowercase, alphanumeric + hyphens, max 63 chars.

### Step 4: Create ConfigMap

```bash
kubectl create configmap ${JOB_NAME}-payload \
  --from-literal=payload="${EVAL_PAYLOAD}" \
  --dry-run=client -o yaml -n {namespace} | kubectl apply -f - -n {namespace}
kubectl label configmap ${JOB_NAME}-payload app=kernel-eval --overwrite -n {namespace}
```

### Step 5: Deploy K8s Job

Render the job template by substituting variables, then apply:

```bash
python3 -c "
import string, sys
tmpl = open(sys.argv[1]).read()
rendered = string.Template(tmpl).safe_substitute(
    JOB_NAME=sys.argv[2],
    BRANCH=sys.argv[3],
    REPO=sys.argv[4],
    VARIANT_ID=sys.argv[5]
)
print(rendered)
" \
  "{job_template_path}" \
  "${JOB_NAME}" \
  "{branch}" \
  "{repo}" \
  "iter-{N}" | kubectl apply -f - -n {namespace}
```

The job template is at the path specified by `evaluator.job_template` in the config (relative to repo root).

### Step 6: Wait for Job completion

Poll every `poll_interval` seconds (from config, default 15) until the job completes, fails, or times out:

```bash
TIMEOUT={timeout}
INTERVAL={poll_interval}
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
  STATUS=$(kubectl get job ${JOB_NAME} -n {namespace} -o jsonpath='{.status.conditions[0].type}' 2>/dev/null)
  if [ "$STATUS" = "Complete" ] || [ "$STATUS" = "Failed" ]; then
    echo "JOB_STATUS:$STATUS"
    break
  fi
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done
if [ $ELAPSED -ge $TIMEOUT ]; then
  echo "JOB_STATUS:Timeout"
fi
```

Use a Bash tool call with `timeout: {timeout * 1000 + 30000}` (timeout in ms, plus buffer).

### Step 7: Collect logs

```bash
kubectl logs job/${JOB_NAME} -c kernel-eval -n {namespace}
```

### Step 8: Parse EVAL_RESULT

Scan the logs for a line containing `EVAL_RESULT:`. The JSON after this prefix contains:

```json
{
  "status": "SUCCESS|COMPILE_ERROR|INCORRECT",
  "fitness": 1.5,
  "error": "",
  "max_diff": 0.0,
  "latency_ms": 2.3,
  "speedup": 1.5,
  "flops": 0.0,
  "compute_ratio": 0.85,
  "memory_transfer_ratio": 0.15,
  "metadata": {}
}
```

Save this JSON to `iteration_{N}/eval_result.json`.

If no `EVAL_RESULT:` line is found, save a synthetic error result:
```json
{
  "status": "COMPILE_ERROR",
  "fitness": 0.0,
  "error": "No EVAL_RESULT in job logs. Last 500 chars of log: ...",
  "latency_ms": 0.0,
  "speedup": 0.0
}
```

### Step 9: Cleanup

Always clean up, even on failure:

```bash
kubectl delete job ${JOB_NAME} -n {namespace} --ignore-not-found
kubectl delete configmap ${JOB_NAME}-payload -n {namespace} --ignore-not-found
```

### Error Handling

- If ConfigMap creation fails: save error result, skip to cleanup
- If Job apply fails: save error result, clean up ConfigMap
- If Job times out: save timeout error result, clean up both
- If log collection fails: save error result with "Failed to collect logs"
- Always run cleanup step regardless of outcome
