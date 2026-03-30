# Profile-Informed Optimization Planning

**Date**: 2026-03-30
**Status**: Approved
**Scope**: Modify `pallas-evolve:start` skill to use raw LLO/HLO/trace profiling data when planning each optimization round

## Problem

The current THINK phase generates optimization variants without seeing the actual compiled IR or hardware utilization data. Sub-agents pick directions based on general TPU knowledge and AGENT.md learnings, but not the specific bottlenecks visible in the kernel's LLO instruction schedule, HLO graph structure, or XPlane trace timeline.

## Solution

Add two new mechanisms to the start skill:

1. **Round 0 (Baseline Profiling)**: Before the first optimization round, submit the unmodified baseline kernel for TPU evaluation to collect its LLO, HLO, and trace artifacts. Commit these to git.

2. **Phase 0 (PROFILE)**: Before each THINK phase, read the previous round's best variant's raw profiling artifacts and generate a `profile_brief.md` that sub-agents consume during variant generation.

## Updated Loop Flow

```
Startup steps 1-9 (unchanged)
  |
Step 10: Baseline profiling (Round 0)
  - Create iteration_0/variants/baseline/kernel.py from template
  - Submit via pallas-evolve:submit (single-variant batch)
  - Download LLO/HLO/trace artifacts
  - Generate baseline/profile_brief.md
  - Commit & push baseline artifacts to GitHub
  |
For each round 1..max_iterations:
  Phase 0: PROFILE (new)
    - Identify prev round's best variant (Round 1: baseline)
    - Read its llo_final.txt, hlo_post_opt.txt, eval_result.json
    - Generate iteration_{N}/profile_brief.md
  Phase 1: THINK (modified)
    - Sub-agents receive profile_brief.md content
    - Must justify approach against profile signals
  Phase 2-6: unchanged
```

## Profile Brief Structure

Generated before each THINK phase from raw artifacts (~500-1000 words):

```markdown
## Profile Brief for Round {N}

### Source
- Kernel: {path} | Speedup: {x}x | Latency: {ms}ms
- Compute ratio: {cr} | Memory transfer ratio: {mtr}

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU | {pct}% | ... |
| Scalar ALU | {pct}% | ... |
| Vector ALU | {pct}% | ... |
| Vector Load/Store | {pct}%/{pct}% | ... |
| Register fills/spills | {n}/{n} | ... |

### Bottleneck Diagnosis
{Multi-signal classification}

### LLO Key Observations
- VLIW bundles: {count} (avg {ops}/bundle)
- MXU dual ratio: {ratio}
- DMA: {count} transfers, {buffering mode}
- {2-3 relevant LLO excerpts, 50-100 lines total}

### HLO Key Observations
- Fusions: {count} | HBM bandwidth: {bytes} ({pct}% of peak)

### Optimization Priorities (derived from profile)
1. {Highest priority based on bottleneck}
2. {Second priority}
3. {Third priority}

### What NOT to try
- {Directions profile data shows won't help}
```

## Modified Sub-Agent Prompt

Each sub-agent receives (additions bolded):

1. Base kernel content
2. AGENT.md learnings
3. Assigned direction
4. TPU v7x rules
5. **Profile brief content**
6. **Direction-specific guidance**: "Given the profile shows {bottleneck}, your direction should focus on {aspect}"

Sub-agents must return:
- Which profile signal motivated their approach
- Expected impact on identified bottleneck
- Which hw utilization metric they expect to improve

Round 2+: sub-agents also receive the delta between current best profile and baseline profile.

## Baseline Directory Structure

```
runs/{kernel}/{run_id}/
  baseline/
    kernel.py
    eval_result.json
    llo_final.txt
    hlo_post_opt.txt
    trace_events.json
    profile_brief.md
```

All baseline artifacts are committed to git and pushed to GitHub for reuse.

## Files Modified

- `skills/start/SKILL.md` — Add step 10 (baseline profiling), Phase 0 (PROFILE), modify Phase 1 (THINK) sub-agent prompts
- No changes to `submit`, `analyze`, or `reflect` skills

## LLO Excerpt Strategy

Raw LLO files can be thousands of lines. The profile brief includes only:
- Inner loop body (between loop markers)
- MXU scheduling section (consecutive `.mxu0`/`.mxu1` ops)
- DMA start/done pairs showing overlap (or lack thereof)
- Register spill patterns (`.vmem_store`/`.vmem_load` pairs)
- Total: 50-100 lines of the most diagnostic sections
