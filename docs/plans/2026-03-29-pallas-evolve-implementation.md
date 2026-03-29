# Pallas-Evolve Skill Group Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `pallas-evolve` Claude Code skill group (plugin) that replaces the automated MAP-Elites engine with an interactive kernel optimization loop driven by Claude Code.

**Architecture:** 4-skill plugin (`start`, `submit`, `analyze`, `reflect`) backed by simplified Pydantic config, existing evaluator data structures, and kubectl/gh CLI for infrastructure. Skills are markdown prompt files that instruct Claude Code step-by-step.

**Tech Stack:** Claude Code plugin system (`.claude-plugin/plugin.json` + `skills/*/SKILL.md`), Python 3.10+ (Pydantic config), kubectl, gh CLI, base64, jq.

**Design doc:** `docs/plans/2026-03-29-pallas-evolve-skill-design.md`

---

### Task 1: Simplify config.py

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/config.py`
- Test: `kernel-evolve/tests/test_config.py`

**Step 1: Write updated tests for simplified config**

Replace `kernel-evolve/tests/test_config.py` with:

```python
"""Tests for simplified YAML config parsing and validation."""

from pathlib import Path

import pytest

from kernel_evolve.config import EvolveConfig, SessionConfig, load_config


@pytest.fixture
def example_config_path():
  return Path(__file__).parent.parent / "examples" / "matmul.yaml"


def test_load_config_from_yaml(example_config_path):
  cfg = load_config(example_config_path)
  assert cfg.kernel.name == "tiled_matmul"
  assert cfg.kernel.evolve_markers.start == "# EVOLVE-BLOCK-START"
  assert len(cfg.shapes) == 2
  assert cfg.shapes[0]["M"] == 1024
  assert cfg.correctness.method == "allclose"
  assert cfg.correctness.rtol == pytest.approx(1e-2)
  assert cfg.tpu.cluster == "tpu7x-cluster"
  assert cfg.session.max_iterations == 20
  assert cfg.session.output_dir == "runs/matmul"


def test_config_defaults():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64, "N": 64, "K": 64}],
    tpu={"cluster": "c", "zone": "z"},
  )
  assert cfg.correctness.method == "allclose"
  assert cfg.evaluator.namespace == "default"
  assert cfg.evaluator.poll_interval == 15
  assert cfg.evaluator.timeout == 600
  assert cfg.session.max_iterations == 20
  assert cfg.session.output_dir == "runs/default"


def test_config_with_evaluator():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
    evaluator={
      "namespace": "custom-ns",
      "job_template": "custom-template.yaml",
      "repo": "user/repo",
      "branch": "dev",
      "poll_interval": 30,
      "timeout": 1200,
    },
  )
  assert cfg.evaluator.namespace == "custom-ns"
  assert cfg.evaluator.repo == "user/repo"
  assert cfg.evaluator.poll_interval == 30


def test_config_with_session():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
    session={"max_iterations": 50, "output_dir": "runs/custom"},
  )
  assert cfg.session.max_iterations == 50
  assert cfg.session.output_dir == "runs/custom"


def test_session_config_defaults():
  s = SessionConfig()
  assert s.max_iterations == 20
  assert s.output_dir == "runs/default"


def test_config_no_evolution_or_llm_fields():
  """Verify old fields (evolution, llm, logging) are not accepted."""
  with pytest.raises(ValueError):
    EvolveConfig(
      kernel={"name": "t", "template": "k.py", "reference": "r.py"},
      shapes=[{"M": 64}],
      tpu={"cluster": "c", "zone": "z"},
      evolution={"population_size": 25},
    )
```

**Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_config.py -v`
Expected: FAIL (old config.py still has EvolutionConfig, LLMConfig, etc.)

**Step 3: Rewrite config.py**

Replace `kernel-evolve/src/kernel_evolve/config.py` with:

```python
"""YAML config parsing and validation with Pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EvolveMarkers(BaseModel):
  start: str = "# EVOLVE-BLOCK-START"
  end: str = "# EVOLVE-BLOCK-END"


class KernelConfig(BaseModel):
  name: str
  template: str
  reference: str
  evolve_markers: EvolveMarkers = Field(default_factory=EvolveMarkers)


class CorrectnessConfig(BaseModel):
  method: str = "allclose"
  rtol: float = 1e-2
  atol: float = 1e-2


class EvaluatorConfig(BaseModel):
  namespace: str = "default"
  job_template: str = ".github/ci/kernel-eval-job.yaml"
  repo: str = ""
  branch: str = "main"
  poll_interval: int = 15
  timeout: int = 600


class TPUConfig(BaseModel):
  cluster: str
  zone: str


class SessionConfig(BaseModel):
  max_iterations: int = 20
  output_dir: str = "runs/default"


class EvolveConfig(BaseModel):
  model_config = {"extra": "forbid"}

  kernel: KernelConfig
  shapes: list[dict[str, Any]]
  correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
  evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
  tpu: TPUConfig
  session: SessionConfig = Field(default_factory=SessionConfig)


def load_config(path: str | Path) -> EvolveConfig:
  """Load and validate an EvolveConfig from a YAML file."""
  with open(path) as f:
    data = yaml.safe_load(f)
  return EvolveConfig(**data)
```

**Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_config.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/config.py kernel-evolve/tests/test_config.py
git commit -m "refactor(config): simplify for skill-based workflow

Remove EvolutionConfig, LLMConfig, LoggingConfig, EvaluatorType.
Add SessionConfig with max_iterations and output_dir.
Set extra='forbid' to reject unknown fields."
```

---

### Task 2: Update example config

**Files:**
- Modify: `kernel-evolve/examples/matmul.yaml`

**Step 1: Rewrite matmul.yaml to simplified format**

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

evaluator:
  namespace: "default"
  job_template: ".github/ci/kernel-eval-job.yaml"
  repo: "sii-xinglong/Glaucis"
  branch: "main"
  poll_interval: 15
  timeout: 600

tpu:
  cluster: "tpu7x-cluster"
  zone: "us-central1"

session:
  max_iterations: 20
  output_dir: "runs/matmul"
```

**Step 2: Run config test to verify YAML loads correctly**

Run: `cd kernel-evolve && python -m pytest tests/test_config.py::test_load_config_from_yaml -v`
Expected: PASS

**Step 3: Commit**

```bash
git add kernel-evolve/examples/matmul.yaml
git commit -m "refactor(config): simplify matmul.yaml for skill workflow

Remove evolution, llm, tpu details, and logging sections.
Add session block with max_iterations and output_dir."
```

---

### Task 3: Create plugin skeleton + orchestrator skill

**Files:**
- Create: `kernel-evolve/plugins/pallas-evolve/.claude-plugin/plugin.json`
- Create: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`

**Step 1: Create plugin.json**

Create `kernel-evolve/plugins/pallas-evolve/.claude-plugin/plugin.json`:

```json
{
  "name": "pallas-evolve",
  "description": "Interactive Pallas TPU kernel optimization via iterative eval-analyze-reflect loop on GKE TPU v7x",
  "version": "0.1.0",
  "author": {
    "name": "sii-xinglong"
  },
  "repository": "https://github.com/sii-xinglong/Glaucis",
  "license": "Apache-2.0",
  "keywords": ["pallas", "tpu", "kernel", "optimization", "jax"]
}
```

**Step 2: Create orchestrator skill**

Create `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`:

````markdown
---
name: start
description: Use when starting a Pallas kernel optimization session on TPU v7x — reads config, creates tracking Issue, and runs the think-submit-analyze-reflect loop
---

# Pallas Kernel Optimization

Optimize a Pallas TPU kernel through iterative mutation, remote evaluation on GKE TPU v7x, profile analysis, and accumulated learning. You are the optimization brain.

## Arguments

Expects a config YAML path relative to `kernel-evolve/examples/`:

```
/pallas-evolve:start matmul.yaml
```

## Startup

Execute these steps in order:

1. **Read config**: Read the YAML config file at `kernel-evolve/examples/<arg>`. Extract: kernel name, template path, reference path, shapes, correctness thresholds, evaluator settings (namespace, job_template, repo, branch, poll_interval, timeout), session settings (max_iterations, output_dir).

2. **Read kernel template**: Read the kernel file at the template path (relative to config dir). Identify the `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` region — this is what you will optimize.

3. **Read reference kernel**: Read the reference implementation. This is the correctness baseline.

4. **Read AGENT.md**: Read `/AGENT.md` at the repo root (if it exists). This contains accumulated learnings:
   - `## Failure Patterns` — known errors and how to avoid them. **You MUST avoid these.**
   - `## Successful Optimizations` — proven strategies. **Prioritize these.**

5. **Check kubectl**: Run `kubectl cluster-info` to verify connectivity to the GKE cluster.

6. **Ask mode**: Use AskUserQuestion with header "Mode" and two options:
   - "Step-by-step" — pause after each iteration for user review and direction
   - "Autonomous" — run continuously until a termination condition is met

7. **Create GitHub Issue**: Run:
   ```bash
   gh issue create \
     --title "[pallas-evolve] Optimize {kernel_name}" \
     --body "Optimizing {kernel_name} kernel for TPU v7x.\n\nConfig: {config_path}\nShapes: {shapes}\nMax iterations: {max_iterations}" \
     --label pallas-evolve
   ```
   Save the issue number from the output.

8. **Initialize run directory**: Create `kernel-evolve/{output_dir}_{YYYYMMDD_HHMMSS}/`:
   - Copy the config YAML as `config.yaml`
   - Copy the template kernel as `baseline_kernel.py`
   - Set `best_speedup = 0.0`, `consecutive_failures = 0`, `iteration = 0`

## Optimization Loop

For each iteration from 1 to `max_iterations`:

### Phase 1: THINK

Read the current kernel:
- Iteration 1: use the template kernel (baseline)
- Later iterations: use `best_kernel.py` from the run directory (or the last iteration's kernel if no improvement yet)

Review context:
- Re-read AGENT.md failure patterns. What must you avoid?
- Re-read AGENT.md successful optimizations. What strategies should you try?
- Read previous iteration results from `iteration_{N-1}/eval_result.json` and `iteration_{N-1}/analysis.md` if they exist
- Look for trends: is the kernel compute-bound or memory-bound? What changed last time?

Formulate your optimization strategy:
- What specific bottleneck are you targeting?
- What code change will you make?
- What improvement do you expect and why?

Create `iteration_{N}/` directory. Write your strategy to `iteration_{N}/strategy.md` with:
```markdown
## Iteration {N} Strategy

**Target bottleneck**: [compute-bound / memory-bound / correctness / compilation]
**Approach**: [specific optimization technique]
**Expected impact**: [why this should improve performance]
**Changes**: [summary of code changes]
```

Write the mutated kernel to `iteration_{N}/kernel.py`:
- Start from the template kernel (keep everything outside EVOLVE-BLOCK unchanged)
- Replace the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` with your optimized version
- Preserve the function signature (`matmul_kernel(x_ref, y_ref, o_ref)` and `optimized_compute(M, N, K)`)
- Validate: the file must be valid Python (no syntax errors)

### Phase 2: SUBMIT

Invoke `pallas-evolve:submit` via the Skill tool.

### Phase 3: ANALYZE

Invoke `pallas-evolve:analyze` via the Skill tool.

### Phase 4: REFLECT

Invoke `pallas-evolve:reflect` via the Skill tool.

### Phase 5: CONTINUE?

Read `iteration_{N}/eval_result.json` to check results:

- If status is SUCCESS and speedup improved over `best_speedup`:
  - Update `best_speedup`
  - Copy kernel to `best_kernel.py` in run directory
  - Reset `consecutive_failures = 0`
- If status is COMPILE_ERROR or INCORRECT:
  - Increment `consecutive_failures`

**Step-by-step mode**: Present a summary table and ask the user:
```
| Iteration | Status | Speedup | Best | Bottleneck |
|-----------|--------|---------|------|------------|
| {N}       | ...    | ...     | ...  | ...        |
```
Options: "Continue", "Adjust strategy", "Stop"

**Autonomous mode**: Check termination:
- `iteration >= max_iterations` → stop
- `consecutive_failures >= 5` → stop
- Otherwise → continue to next iteration

## End of Session

When the loop terminates:

1. Report final results: best speedup, total iterations, key optimizations that worked
2. If `best_speedup > 1.0`:
   - Create a branch: `git checkout -b pallas-evolve/{kernel_name}-{timestamp}`
   - Copy `best_kernel.py` to the template kernel path
   - Commit and push
   - Create PR:
     ```bash
     gh pr create \
       --title "[pallas-evolve] {kernel_name}: 1.0x → {best_speedup}x" \
       --body "Optimized via pallas-evolve. See #{issue_number} for iteration history."
     ```
3. Close the GitHub Issue with a final summary comment

## TPU v7x Pallas Optimization Knowledge

When writing kernel mutations, follow these TPU v7x Ironwood constraints:

**Hard rules (violating these causes compilation errors):**
- Always use `jnp.bfloat16`, never `jnp.float16` — Mosaic compiler requires it
- Use `Ref` indexing: `x_ref[...]` or `x_ref[pl.ds(start, size)]` — never `pl.load()`
- Use `pl.ds(start, size)` — never `pl.dslice(start, size)`
- Kernel function signature must stay unchanged (same Ref arguments)
- `optimized_compute` signature must stay unchanged (M, N, K with defaults)

**Optimization levers (from most to least impactful):**
1. **K-tiling**: Split the K (reduction) dimension into tiles. Accumulate partial results in scratch memory. Reduces HBM bandwidth pressure.
2. **Block size tuning**: Try 64, 128, 256. Larger blocks = more compute per tile but more VMEM. Match to matrix dimensions.
3. **Scratch memory**: Use `pltpu.SemaphoreType.REGULAR` scratch for fast VMEM accumulators instead of writing back to HBM each tile.
4. **Pipelining**: Use `pltpu.emit_pipeline` to overlap compute and memory transfers across tiles.
5. **Vectorization**: Ensure inner dimensions are multiples of 128 for MXU efficiency.

**Common pitfalls:**
- Block size 512 may OOM on 2048x2048 matrices (VMEM limit)
- Accumulator dtype should be `jnp.float32` for numerical stability, cast to `bfloat16` on store
- Grid dimensions must evenly divide matrix dimensions
````

**Step 3: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/
git commit -m "feat(skills): add pallas-evolve plugin skeleton and orchestrator skill"
```

---

### Task 4: Write submit skill

**Files:**
- Create: `kernel-evolve/plugins/pallas-evolve/skills/submit/SKILL.md`

**Step 1: Create submit skill**

Create `kernel-evolve/plugins/pallas-evolve/skills/submit/SKILL.md`:

````markdown
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
````

**Step 2: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/submit/SKILL.md
git commit -m "feat(skills): add pallas-evolve:submit skill for kubectl eval"
```

---

### Task 5: Write analyze skill

**Files:**
- Create: `kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md`

**Step 1: Create analyze skill**

Create `kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md`:

````markdown
---
name: analyze
description: Use when analyzing TPU kernel evaluation results — parses eval_result.json, classifies bottlenecks, compares with history, writes analysis.md
---

# Analyze Evaluation Results

Parse the evaluation result from a TPU kernel run, classify performance bottlenecks, compare with previous iterations, and write a structured analysis.

## Context

Invoked by `pallas-evolve:start` after `pallas-evolve:submit`, or standalone for debugging. Expects `iteration_{N}/eval_result.json` to exist in the run directory.

## Procedure

### Step 1: Read eval result

Read `iteration_{N}/eval_result.json`. The JSON has this structure:

```json
{
  "status": "SUCCESS|COMPILE_ERROR|INCORRECT",
  "fitness": 1.5,
  "error": "error message if failed",
  "max_diff": 0.0,
  "latency_ms": 2.3,
  "speedup": 1.5,
  "compute_ratio": 0.85,
  "memory_transfer_ratio": 0.15,
  "metadata": {}
}
```

### Step 2: Classify result

**If status is COMPILE_ERROR:**
- Report the error message
- Identify the likely cause:
  - `SyntaxError` → Python syntax issue in kernel code
  - `TypeError` / `ValueError` → Wrong argument types or shapes
  - `Mosaic` or `MLIR` in error → TPU compiler issue (likely dtype or API misuse)
  - `ResourceExhausted` → VMEM/memory overflow (block size too large)
- Suggest specific fix based on the error
- Write to analysis.md and return

**If status is INCORRECT:**
- Report `max_diff` (maximum absolute difference from reference)
- Compare with correctness thresholds (`rtol`, `atol` from config)
- Common causes: integer overflow, wrong accumulator dtype, incorrect tiling boundaries
- Write to analysis.md and return

**If status is SUCCESS:**
- Proceed to performance analysis (Step 3)

### Step 3: Performance analysis (SUCCESS only)

Extract key metrics:
- `speedup`: ratio of reference_latency / kernel_latency. >1.0 means faster than baseline.
- `latency_ms`: absolute kernel latency
- `compute_ratio`: fraction of time spent on useful computation (0.0-1.0). Higher is better.
- `memory_transfer_ratio`: fraction of time spent on memory transfers / sync waits (0.0-1.0). Lower is better.

**Bottleneck classification:**

| compute_ratio | Classification | Primary Bottleneck |
|---------------|---------------|-------------------|
| >= 0.75       | Compute-bound | MXU utilization, vectorization |
| 0.50 - 0.75   | Balanced      | Both compute and memory |
| < 0.50        | Memory-bound  | HBM bandwidth, data movement |
| None          | Unknown       | Profiling data not available |

### Step 4: Trend analysis

Read previous iteration results (if any) from `iteration_{N-1}/eval_result.json`, `iteration_{N-2}/eval_result.json`, etc.:

- **Speedup trend**: improving, flat, or regressing?
- **compute_ratio trend**: are we becoming more compute-efficient?
- **Strategy correlation**: which optimization approaches improved things?

Flag any regressions (speedup decreased from previous iteration).

### Step 5: Generate optimization suggestions

Based on the bottleneck classification, suggest next steps:

**Memory-bound (compute_ratio < 0.50):**
- Increase block size to process more data per tile
- Add K-tiling to reduce HBM round-trips
- Use scratch memory for accumulators
- Add pipelining to overlap compute and memory

**Compute-bound (compute_ratio >= 0.75):**
- Optimize inner loop vectorization
- Ensure dimensions are multiples of 128 for MXU
- Try different block aspect ratios
- Consider algorithmic improvements

**Balanced (0.50 - 0.75):**
- Profile deeper: which specific operations are slow?
- Try pipelining to overlap compute and memory
- Adjust block sizes to find the sweet spot

**Regression detected:**
- Compare the current kernel with the previous best
- Identify what changed and why it was slower
- Suggest reverting the specific change that caused regression

### Step 6: Write analysis

Write `iteration_{N}/analysis.md`:

```markdown
## Iteration {N} Analysis

**Status**: {SUCCESS/COMPILE_ERROR/INCORRECT}
**Speedup**: {speedup}x (best so far: {best_speedup}x)
**Latency**: {latency_ms}ms
**Compute ratio**: {compute_ratio} ({classification})

### Bottleneck
{Description of the primary bottleneck}

### Trend
{Comparison with previous iterations}

### Suggestions
{Specific optimization suggestions for next iteration}
```
````

**Step 2: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md
git commit -m "feat(skills): add pallas-evolve:analyze skill for profile analysis"
```

---

### Task 6: Write reflect skill

**Files:**
- Create: `kernel-evolve/plugins/pallas-evolve/skills/reflect/SKILL.md`

**Step 1: Create reflect skill**

Create `kernel-evolve/plugins/pallas-evolve/skills/reflect/SKILL.md`:

````markdown
---
name: reflect
description: Use when recording kernel optimization learnings to AGENT.md — extracts failure patterns and successful optimization root causes, updates GitHub Issue
---

# Record Optimization Learnings

After each evaluation iteration, extract learnings and record them to AGENT.md. Update the GitHub Issue with iteration results.

## Context

Invoked by `pallas-evolve:start` after `pallas-evolve:analyze`, or standalone. Expects both `iteration_{N}/eval_result.json` and `iteration_{N}/analysis.md` to exist.

## Procedure

### Step 1: Read iteration data

Read:
- `iteration_{N}/eval_result.json` — the raw evaluation result
- `iteration_{N}/analysis.md` — the bottleneck analysis
- `iteration_{N}/strategy.md` — what optimization was attempted
- `AGENT.md` at the repo root — existing learnings

### Step 2: Determine if this is a new learning

Not every iteration produces a learning. Only record when:

**Failure worth recording** (new pattern not already in AGENT.md):
- A compilation error with a non-obvious cause
- A correctness error that reveals a Pallas/TPU constraint
- A timeout or infrastructure failure with a reproducible cause

**Success worth recording** (meaningful improvement):
- Speedup improved over the previous best
- A specific technique produced a measurable improvement
- The improvement reveals a generalizable optimization principle

Skip recording if:
- The failure is a trivial syntax error (typo, missing import)
- The speedup change is negligible (<5% improvement)
- A similar pattern already exists in AGENT.md (check by reading existing entries)

### Step 3: Update AGENT.md (if new learning)

AGENT.md lives at the **repo root** (`/AGENT.md`). If it doesn't exist, create it with:

```markdown
# Pallas Kernel Optimization Agent Knowledge

## Failure Patterns

## Successful Optimizations
```

**For failures**, find the next available `[Fxxx]` number and append under `## Failure Patterns`:

```markdown
### [F{NNN}] {Short description}
- **Symptom**: {What the error looks like — include key error message text}
- **Root cause**: {Why it happens — the underlying Pallas/TPU constraint}
- **Fix**: {How to avoid it in future kernels}
- **First seen**: {YYYY-MM-DD}, {kernel_name} optimization
```

**For successful optimizations**, find the next `[Sxxx]` number and append under `## Successful Optimizations`:

```markdown
### [S{NNN}] {Short description of the technique}
- **Optimization**: {What was changed in the kernel code}
- **Impact**: {before_speedup}x -> {after_speedup}x on {shape}
- **Why it works**: {Root cause analysis — why this specific change improved performance}
- **Applicable when**: {Conditions where this technique should be tried again}
- **First seen**: {YYYY-MM-DD}, {kernel_name} optimization
```

**Deduplication**: Before adding, read all existing entries. If a similar pattern exists:
- For failures: update the existing entry with additional context rather than creating a duplicate
- For successes: update with the new performance data if the technique is the same

### Step 4: Comment on GitHub Issue

Post an iteration summary comment:

```bash
gh issue comment {issue_number} --body "$(cat <<'EOF'
### Iteration {N}

**Status**: {SUCCESS/COMPILE_ERROR/INCORRECT}
**Speedup**: {speedup}x (best: {best_speedup}x)
**Strategy**: {one-line summary from strategy.md}
**Bottleneck**: {classification from analysis.md}

{If AGENT.md was updated: "Recorded learning: [F/S{NNN}] {description}"}
EOF
)"
```

### Step 5: Commit AGENT.md changes (if any)

If AGENT.md was modified:

```bash
git add AGENT.md
git commit -m "docs(agent): record {F/S}{NNN} from {kernel_name} iteration {N}"
```
````

**Step 2: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/reflect/SKILL.md
git commit -m "feat(skills): add pallas-evolve:reflect skill for AGENT.md learnings"
```

---

### Task 7: Create AGENT.md template

**Files:**
- Create: `AGENT.md` (repo root)

**Step 1: Create initial AGENT.md**

Create `AGENT.md` at the repo root:

```markdown
# Pallas Kernel Optimization Agent Knowledge

## Failure Patterns

## Successful Optimizations
```

**Step 2: Commit**

```bash
git add AGENT.md
git commit -m "docs: add AGENT.md template for optimization learnings"
```

---

### Task 8: Remove deprecated code and tests

**Files:**
- Delete: `kernel-evolve/src/kernel_evolve/engine.py`
- Delete: `kernel-evolve/src/kernel_evolve/population.py`
- Delete: `kernel-evolve/src/kernel_evolve/cli.py`
- Delete: `kernel-evolve/src/kernel_evolve/ci_dispatcher.py`
- Delete: `kernel-evolve/src/kernel_evolve/perf_log.py`
- Delete: `kernel-evolve/src/kernel_evolve/llm/` (entire directory)
- Delete: `kernel-evolve/tests/test_engine.py`
- Delete: `kernel-evolve/tests/test_population.py`
- Delete: `kernel-evolve/tests/test_cli.py`
- Delete: `kernel-evolve/tests/test_ci_dispatcher.py`
- Delete: `kernel-evolve/tests/test_perf_log.py`
- Delete: `kernel-evolve/tests/test_llm.py`
- Delete: `kernel-evolve/tests/test_llm_providers.py`
- Delete: `kernel-evolve/tests/test_integration.py`

**Step 1: Delete deprecated source files**

```bash
cd kernel-evolve
rm -f src/kernel_evolve/engine.py
rm -f src/kernel_evolve/population.py
rm -f src/kernel_evolve/cli.py
rm -f src/kernel_evolve/ci_dispatcher.py
rm -f src/kernel_evolve/perf_log.py
rm -rf src/kernel_evolve/llm/
```

**Step 2: Delete deprecated test files**

```bash
rm -f tests/test_engine.py
rm -f tests/test_population.py
rm -f tests/test_cli.py
rm -f tests/test_ci_dispatcher.py
rm -f tests/test_perf_log.py
rm -f tests/test_llm.py
rm -f tests/test_llm_providers.py
rm -f tests/test_integration.py
```

**Step 3: Clean up __init__.py imports**

Read `kernel-evolve/src/kernel_evolve/__init__.py` and remove any imports of deleted modules. If it's empty or only has version info, leave it as-is.

**Step 4: Run remaining tests to verify nothing is broken**

Run: `cd kernel-evolve && python -m pytest tests/ -v`
Expected: All remaining tests pass (test_config, test_evaluator, test_mutation, test_kube_evaluator, test_profiler)

**Step 5: Commit**

```bash
git add -A kernel-evolve/src/kernel_evolve/ kernel-evolve/tests/
git commit -m "refactor: remove MAP-Elites engine, LLM providers, and CLI

Replaced by pallas-evolve skill group. Kept: evaluator, mutation,
kube_evaluator, profiler, config (simplified)."
```

---

### Task 9: Update pyproject.toml

**Files:**
- Modify: `kernel-evolve/pyproject.toml`

**Step 1: Read current pyproject.toml**

Read `kernel-evolve/pyproject.toml` to understand current structure.

**Step 2: Update pyproject.toml**

Make these changes:
- Remove `click` from core dependencies
- Remove the `[project.scripts]` entry point (`kernel-evolve = kernel_evolve.cli:main`)
- Remove `anthropic`, `google-genai`, `openai` from optional dependencies
- Keep: `pyyaml`, `pydantic` (core), `matplotlib` (charts), `xprof` (profile), `pytest`/`ruff` (dev)

**Step 3: Verify package still installs**

Run: `cd kernel-evolve && pip install -e ".[dev]"` (or `uv pip install -e ".[dev]"`)
Expected: Installs without errors

**Step 4: Run tests**

Run: `cd kernel-evolve && python -m pytest tests/ -v`
Expected: All remaining tests PASS

**Step 5: Commit**

```bash
git add kernel-evolve/pyproject.toml
git commit -m "refactor(deps): remove click, LLM providers from dependencies

CLI entry point removed (replaced by skill invocation).
LLM providers removed (Claude Code is the optimizer now)."
```

---

### Task 10: Plugin registration and verification

**Step 1: Verify plugin structure**

Run: `find kernel-evolve/plugins/pallas-evolve -type f | sort`

Expected output:
```
kernel-evolve/plugins/pallas-evolve/.claude-plugin/plugin.json
kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md
kernel-evolve/plugins/pallas-evolve/skills/reflect/SKILL.md
kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
kernel-evolve/plugins/pallas-evolve/skills/submit/SKILL.md
```

**Step 2: Install plugin locally**

Symlink the plugin to the Claude Code local plugins directory:

```bash
mkdir -p ~/.claude/plugins/cache/local/pallas-evolve/0.1.0
ln -sf "$(pwd)/kernel-evolve/plugins/pallas-evolve/.claude-plugin" ~/.claude/plugins/cache/local/pallas-evolve/0.1.0/.claude-plugin
ln -sf "$(pwd)/kernel-evolve/plugins/pallas-evolve/skills" ~/.claude/plugins/cache/local/pallas-evolve/0.1.0/skills
```

Then add to `~/.claude/settings.json` under `enabledPlugins`:
```json
"pallas-evolve@local": true
```

**Step 3: Verify skills are discoverable**

Start a new Claude Code session and check that `/pallas-evolve:start`, `/pallas-evolve:submit`, `/pallas-evolve:analyze`, and `/pallas-evolve:reflect` are recognized.

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete pallas-evolve skill group implementation

Replaces the automated MAP-Elites kernel evolution engine with an
interactive Claude Code skill group for iterative Pallas TPU kernel
optimization on GKE TPU v7x."
```
