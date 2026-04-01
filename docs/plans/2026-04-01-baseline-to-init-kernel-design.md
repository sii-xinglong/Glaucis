# Move Baseline Profiling from start to init-kernel

## Summary

Move baseline evaluation and profile generation from `pallas-evolve:start` Step 10 into `pallas-evolve:init-kernel` Step 9d (now mandatory). Extract profile brief generation into a new `pallas-evolve:profile-brief` skill. Baseline artifacts become permanent project files in `examples/kernels/{KERNEL_NAME}_baseline/`.

## Motivation

Currently, `init-kernel` generates kernel files but leaves baseline profiling to `start`. This means:
- `start` must always run a Round 0 baseline before beginning optimization
- The baseline is tied to a specific run directory, not the project
- Profile brief generation logic is embedded inline in `start`, making it non-reusable

After this change, `init-kernel` fully bootstraps a kernel project (code + verified baseline), and `start` focuses purely on the optimization loop.

## Design

### 1. New skill: `pallas-evolve:profile-brief`

**Path**: `kernel-evolve/plugins/pallas-evolve/skills/profile-brief/SKILL.md`

**Arguments**:
```
/pallas-evolve:profile-brief <artifacts_dir> [--baseline <baseline_dir>] [--round <N>]
```

- `artifacts_dir`: directory containing `eval_result.json` (and optionally `llo_final.txt`, `hlo_post_opt.txt`, `trace_events.json`)
- `--baseline`: optional baseline directory for delta comparison table (Round 2+)
- `--round`: round number for the header (default: 0)

**Content**: The full "Profile Brief Generation" procedure currently in `start` lines 93-214, including:
- Reading `eval_result.json` and extracting metrics (speedup, latency, compute_ratio, hw_utilization, deep profiling)
- Reading optional LLO/HLO/trace files and extracting key excerpts
- Bottleneck classification using the multi-signal table
- Deriving optimization priorities (top 3, ranked)
- Identifying what NOT to try (profile-evidence based)
- Writing `profile_brief.md` using the structured template
- Delta vs baseline table (when `--baseline` is provided)

Also includes the deep profiling signal descriptions currently at the bottom of `start` (lines 476-485).

### 2. Changes to `init-kernel`

**Step 9d — Baseline profiling (Round 0)** becomes mandatory (replaces optional TPU verification):

1. Verify kubectl connectivity. If not connected, stop with error.
2. Create baseline directory: `kernel-evolve/examples/kernels/{KERNEL_NAME}_baseline/`
3. Copy template as `kernel.py` into baseline directory.
4. Set up temporary iteration structure for `submit` skill, invoke `pallas-evolve:submit`, copy results to `_baseline/` directory.
5. If result is `COMPILE_ERROR`, stop — template kernel is broken.
6. Invoke `pallas-evolve:profile-brief` on the baseline directory with `--round 0`.
7. Commit baseline artifacts.

**Step 10 summary** updated to list `kernel-evolve/examples/kernels/{KERNEL_NAME}_baseline/` in generated files.

### 3. Changes to `start`

**Remove**:
- Step 10 ("Baseline profiling (Round 0)") — entirely removed
- "Profile Brief Generation" section — moved to `profile-brief` skill
- Deep profiling signals section (lines 476-485) — moved to `profile-brief` skill

**Modify Step 6** (Check kubectl): Also verify baseline exists:
```bash
ls kernel-evolve/examples/kernels/{KERNEL_NAME}_baseline/eval_result.json
```
If missing, stop: "Baseline not found. Run /pallas-evolve:init-kernel first."

**Modify Step 9** (Initialize run directory): Copy baseline into run dir:
```
cp -r kernel-evolve/examples/kernels/{KERNEL_NAME}_baseline/ {run_dir}/baseline/
```

**Modify Phase 0**:
- Round 1: Read baseline profile brief from `{run_dir}/baseline/profile_brief.md` (already exists)
- Round 2+: Invoke `pallas-evolve:profile-brief` on current best variant's directory with `--baseline {run_dir}/baseline/ --round {N}`

**Keep**: "TPU v7x Pallas Optimization Knowledge" section (hard rules, optimization levers, common pitfalls) — used by variant generation sub-agents.

## File Changes

| File | Action |
|------|--------|
| `kernel-evolve/plugins/pallas-evolve/skills/profile-brief/SKILL.md` | Create (new skill) |
| `kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md` | Modify Step 9d, Step 10 |
| `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md` | Remove Step 10, Profile Brief section, deep profiling signals; modify Steps 6, 9, Phase 0 |
