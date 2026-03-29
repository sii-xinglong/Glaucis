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
