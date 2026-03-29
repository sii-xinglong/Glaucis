# Pallas-Evolve Skill Group Design

**Date**: 2026-03-29
**Status**: Approved
**Scope**: Replace the automated MAP-Elites kernel evolution engine with an interactive Claude Code skill group

## Context

The current `kernel-evolve` system is an automated MAP-Elites evolution engine that uses LLM mutation prompts to optimize Pallas TPU kernels. It dispatches evaluations to GKE TPU v7x via kubectl, collects profiling data, and manages a population archive.

This design replaces that automated loop with a **skill group** (`pallas-evolve`) where Claude Code acts as the optimization brain — reading kernels, reasoning about optimizations, submitting evaluations, analyzing profiles, and accumulating learnings.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Scope | Full replacement of MAP-Elites engine |
| Control mode | Both step-by-step and autonomous, user chooses at startup |
| AGENT.md | Single file at repo root; only distilled failure/success reflections |
| State tracking | GitHub Issues/PRs for session tracking; run directories for per-run data |
| Eval path | Direct kubectl (KubeEvaluator pattern) |
| Skill structure | Orchestrator + utility skills (Approach B) |
| Skill group name | `pallas-evolve` |

## Architecture

### Skill Group

```
pallas-evolve           (orchestrator — entry point, loop control, optimization thinking)
├── pallas-evolve:submit   (kubectl Job submission, polling, result collection)
├── pallas-evolve:analyze  (profile parsing, bottleneck identification, trend analysis)
└── pallas-evolve:reflect  (AGENT.md updates, GitHub Issue updates)
```

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  /pallas-evolve matmul.yaml                             │
│                                                         │
│  ┌──────────┐                                           │
│  │  START    │ Read config, kernel, AGENT.md             │
│  │          │ Create GitHub Issue                        │
│  │          │ Ask: step-by-step or autonomous?           │
│  └────┬─────┘                                           │
│       ▼                                                 │
│  ┌──────────┐                                           │
│  │  THINK   │ Analyze kernel + AGENT.md + past results  │
│  │          │ Propose optimization strategy              │
│  │          │ Write mutated kernel to run dir            │
│  └────┬─────┘◄────────────────────────────┐             │
│       ▼                                   │             │
│  ┌──────────┐                             │             │
│  │  SUBMIT  │ /pallas-evolve:submit       │             │
│  │          │ kubectl Job → TPU eval      │             │
│  │          │ Collect EVAL_RESULT          │             │
│  └────┬─────┘                             │             │
│       ▼                                   │             │
│  ┌──────────┐                             │             │
│  │ ANALYZE  │ /pallas-evolve:analyze      │             │
│  │          │ Parse profile data           │             │
│  │          │ Identify bottlenecks         │             │
│  └────┬─────┘                             │             │
│       ▼                                   │             │
│  ┌──────────┐                             │             │
│  │ REFLECT  │ /pallas-evolve:reflect      │             │
│  │          │ Update AGENT.md             │             │
│  │          │ Update GitHub Issue          │             │
│  └────┬─────┘                             │             │
│       ▼                                   │             │
│  ┌──────────┐      ┌───┐                  │             │
│  │ CONTINUE?│─yes──► ● │─────────────────┘              │
│  │          │       └───┘                               │
│  │          │─no───► Output best kernel + summary PR    │
│  └──────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

### Loop Termination

- User interrupt (step-by-step: user says stop; autonomous: Ctrl-C)
- Max iterations reached (from config `session.max_iterations`)
- Consecutive failures exceed threshold (5 in a row)

### End of Session

- Save best kernel to `runs/{session}/best_kernel.py`
- Create summary PR with best kernel if it improved over baseline
- Close or update the GitHub Issue with final results

## Skill Details

### `pallas-evolve` (Orchestrator)

**Invocation**: `/pallas-evolve <config.yaml>`

**Responsibilities**:
1. Parse config YAML path from arguments
2. Load config, kernel template, reference kernel, AGENT.md
3. Ask mode: step-by-step or autonomous (via AskUserQuestion)
4. Create GitHub Issue: `gh issue create --title "[pallas-evolve] Optimize {name}"`
5. Initialize run directory: `runs/{name}_{timestamp}/`
6. Enter optimization loop:
   - Analyze current kernel (first iteration: template; later: last iteration's kernel)
   - Read AGENT.md failure patterns to avoid known pitfalls
   - Read past iteration results for trend awareness
   - Propose optimization strategy, write to `iteration_{N}/strategy.md`
   - Write mutated kernel using EVOLVE-BLOCK extract/inject
   - Invoke `pallas-evolve:submit`
   - Invoke `pallas-evolve:analyze`
   - Invoke `pallas-evolve:reflect`
   - Step-by-step: present summary, ask user to continue
   - Autonomous: check termination conditions, loop
7. On exit: save best kernel, update Issue, optionally create PR

### `pallas-evolve:submit` (Eval Submission)

**Invocation**: `/pallas-evolve:submit` (standalone debug) or invoked by orchestrator

**Responsibilities**:
1. Read current kernel from run directory (latest `iteration_{N}/kernel.py`)
2. Read reference kernel and shapes from config
3. Construct EvalRequest, base64-encode it
4. Create ConfigMap: `kubectl create configmap {name}-payload --from-literal=EVAL_PAYLOAD={b64}`
5. Render K8s Job template (substitute `$JOB_NAME`, `$REPO`, `$BRANCH`, `$VARIANT_ID`)
6. Apply Job: `kubectl apply -f -`
7. Poll job status: `kubectl get job {name} -o jsonpath='{.status.conditions[*].type}'` every 15s
8. On complete: `kubectl logs job/{name} -c kernel-eval`
9. Parse `EVAL_RESULT:{json}` from logs
10. Save to `iteration_{N}/eval_result.json`
11. Cleanup: `kubectl delete job/{name} && kubectl delete configmap/{name}-payload`

### `pallas-evolve:analyze` (Profile Analysis)

**Invocation**: `/pallas-evolve:analyze` or invoked by orchestrator

**Responsibilities**:
1. Read `iteration_{N}/eval_result.json`
2. If eval failed: report error type and message, suggest what went wrong
3. If eval succeeded with profile data:
   - Parse `compute_ratio` and `memory_transfer_ratio`
   - Classify: compute-bound (ratio > 0.75), memory-bound (ratio < 0.5), or balanced
   - Compare with previous iterations to detect trends
4. Identify specific bottlenecks and suggest optimization directions:
   - Memory-bound → suggest: larger blocks, scratch memory, K-tiling
   - Compute-bound → suggest: pipeline stages, vectorization
   - Regression detected → flag what changed
5. Write analysis to `iteration_{N}/analysis.md`

### `pallas-evolve:reflect` (Learning Recorder)

**Invocation**: `/pallas-evolve:reflect` or invoked by orchestrator

**Responsibilities**:
1. Read current iteration's `eval_result.json` and `analysis.md`
2. Read existing AGENT.md
3. **On failure**:
   - Extract root cause and fix pattern
   - Check for duplicate: does a similar [Fxxx] entry exist?
   - If new: append under `## Failure Patterns` with next number
   - If existing: update the entry with additional context
4. **On successful optimization** (speedup improved):
   - Extract why the optimization worked
   - Check for duplicate [Sxxx] entry
   - If new: append under `## Successful Optimizations`
5. Update GitHub Issue with iteration summary comment:
   - `gh issue comment {issue_number} --body "Iteration {N}: {status}, speedup={x}"`

## AGENT.md Structure

Location: repo root `/AGENT.md`

```markdown
# Pallas Kernel Optimization Agent Knowledge

## Failure Patterns

### [F001] Short description
- **Symptom**: What the error looks like
- **Root cause**: Why it happens
- **Fix**: How to avoid it
- **First seen**: Date, kernel name

## Successful Optimizations

### [S001] Short description
- **Optimization**: What was changed
- **Impact**: Before → after speedup
- **Why it works**: Root cause analysis
- **Applicable when**: When to reuse this pattern
- **First seen**: Date, kernel name
```

## Per-Iteration Directory Structure

```
runs/{name}_{timestamp}/
├── config.yaml             # Copy of the config used
├── baseline_kernel.py      # Original template kernel
├── best_kernel.py          # Best kernel found so far (updated each iteration)
├── iteration_1/
│   ├── kernel.py           # Mutated kernel for this iteration
│   ├── strategy.md         # Optimization reasoning (what and why)
│   ├── eval_result.json    # Raw eval output
│   └── analysis.md         # Bottleneck analysis
├── iteration_2/
│   ├── ...
└── iteration_N/
    └── ...
```

## Config Format

Simplified from the current `matmul.yaml`. Removed: `LLMConfig`, `EvolutionConfig`, `population` fields. Added: `session` block.

```yaml
kernel:
  name: tiled_matmul
  template: kernels/matmul.py
  reference: kernels/matmul_ref.py

shapes:
  - [1024, 1024, 1024]
  - [2048, 2048, 2048]

correctness:
  method: allclose
  rtol: 1e-2
  atol: 1.0

evaluator:
  namespace: default
  job_template: .github/ci/kernel-eval-job.yaml
  repo: sii-xinglong/Glaucis
  branch: main

tpu:
  cluster: tpu7x-cluster
  zone: us-central1

session:
  max_iterations: 20
  output_dir: runs/matmul
```

## Skill File Layout

```
kernel-evolve/skills/pallas-evolve/
├── pallas-evolve.md          # Orchestrator (entry point)
├── submit.md                 # kubectl eval submission
├── analyze.md                # Profile analysis
└── reflect.md                # AGENT.md learning recorder
```

## Infrastructure Changes

### Keep (reused by skills via Claude Code tools)

| File | Role in skill system |
|------|---------------------|
| `evaluator.py` | `EvalRequest`/`EvalResult` data structures, `encode_b64`/`decode_b64` — referenced by submit skill for payload format |
| `config.py` | Pydantic config models (simplified to remove LLM/evolution fields) |
| `mutation.py` | EVOLVE-BLOCK extract/inject — Claude uses this logic for kernel manipulation |
| `docker/evaluate.py` | Runs inside TPU pod unchanged |
| `.github/ci/kernel-eval-job.yaml` | K8s Job template unchanged |
| `profiler.py` | Profile analysis utilities — referenced by analyze skill |

### Remove

| File | Reason |
|------|--------|
| `engine.py` | MAP-Elites loop replaced by skill orchestration |
| `population.py` | Archive/Variant/BehaviorDescriptor no longer needed |
| `llm/` (entire directory) | Claude Code is the LLM now |
| `cli.py` | Click CLI replaced by skill invocation |
| `ci_dispatcher.py` | GitHub Actions eval path removed (kubectl only) |
| `perf_log.py` | Replaced by per-iteration `analysis.md` files |

### Modify

| File | Changes |
|------|---------|
| `config.py` | Remove `LLMConfig`, `EvolutionConfig`, `EvaluatorConfig.type`; add `SessionConfig` |
| `pyproject.toml` | Remove `click`, LLM provider deps from core; remove CLI entry point |

## GitHub Integration

### Issue Tracking

Each optimization session creates a GitHub Issue:
- **Title**: `[pallas-evolve] Optimize {kernel_name}`
- **Body**: Config summary, baseline kernel info, optimization goals
- **Labels**: `pallas-evolve`
- **Comments**: One per iteration with results summary
- **Closed**: When session ends (with final summary)

### PR Creation

When the session finds an improvement:
- **Branch**: `pallas-evolve/{kernel_name}-{timestamp}`
- **Title**: `[pallas-evolve] {kernel_name}: {baseline_speedup}x → {best_speedup}x`
- **Body**: Summary of optimizations, link to tracking Issue
- **Files**: Updated kernel, run artifacts
