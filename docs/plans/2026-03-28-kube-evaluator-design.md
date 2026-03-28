# KubeEvaluator: Direct kubectl-based TPU Kernel Evaluation

**Date:** 2026-03-28
**Status:** Approved

## Problem

The kernel-evolve optimizer only supports evaluation via GitHub Actions CI (`CIDispatcher`). This adds unnecessary latency (CI queue time, GitHub API overhead) and prevents local development iteration. We need a direct evaluator that submits K8s Jobs to the GKE TPU cluster.

## Solution

Create `KubeEvaluator` that implements the `Evaluator` interface and directly creates K8s Jobs on `tpu7x-cluster` via `kubectl`. Reuse the existing `kernel-eval-job.yaml` template and `evaluate.py` script.

## Architecture

```
CLI (local) --> Claude API (opus-4.6) --> MutationResponse
    |
    v
EvolutionEngine (MAP-Elites) --> KubeEvaluator
                                   |  1. Create ConfigMap (eval payload)
                                   |  2. Render & apply Job from template
                                   |  3. Poll job status until complete
                                   |  4. Read pod logs -> EvalResult
                                   |  5. Cleanup Job + ConfigMap
                                   v
                              GKE tpu7x-cluster
                              TPU v7x (2x2x1)
                              evaluate.py (3-stage eval)
```

## Components

### 1. KubeEvaluator (`kube_evaluator.py`)

- Implements `Evaluator.evaluate(EvalRequest) -> EvalResult`
- Config: namespace, job template path, repo, branch, poll interval, timeout
- Each evaluation:
  1. Generate unique job name from variant ID
  2. Create ConfigMap with base64 eval payload
  3. Render job template (envsubst-style: `$JOB_NAME`, `$BRANCH`, `$REPO`, `$VARIANT_ID`)
  4. `kubectl apply` the rendered YAML
  5. Poll job status every 15s up to timeout
  6. On completion: read pod logs, extract `EVAL_RESULT:` JSON
  7. Cleanup: delete Job and ConfigMap

### 2. Config Changes (`config.py`)

- Add `evaluator` field to `EvolveConfig` with enum `ci | kube` (default: `kube`)
- Add `KubeConfig` model: namespace, job_template, repo, branch, poll_interval, timeout

### 3. CLI Changes (`cli.py`)

- Route to `KubeEvaluator` or `CIDispatcher` based on `evaluator` config
- Default to `kube`

### 4. matmul.yaml Updates

- Set `evaluator: kube`
- Update TPU section: cluster=`tpu7x-cluster`, zone=`us-central1`
- Set repo to `sii-xinglong/Glaucis`
- Keep: Claude opus-4.6, 3 islands, 50 generations

## Error Handling

- Pod scheduling timeout -> `EvalResult.compile_error("Pod scheduling timeout")`
- Job failure -> read logs for error details
- No `EVAL_RESULT:` in logs -> `EvalResult.compile_error("No result in logs")`
- kubectl errors -> fail with descriptive error

## Prerequisites

- `kubectl` configured with access to `tpu7x-cluster`
- `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` env var set
- TPU v7x nodepool available in cluster
