# Move Baseline Profiling to init-kernel — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move baseline evaluation from `start` to `init-kernel`, extract profile brief generation into a new `profile-brief` skill.

**Architecture:** Three skill files change: (1) new `profile-brief` skill gets the Profile Brief Generation procedure and deep profiling signals from `start`, (2) `init-kernel` Step 9d becomes mandatory baseline profiling that invokes `submit` + `profile-brief`, (3) `start` removes Step 10/Profile Brief/deep profiling, adds baseline existence check and uses `profile-brief` skill in Phase 0.

**Tech Stack:** Claude Code skills (SKILL.md markdown files), pallas-evolve plugin system

---

### Task 1: Create `pallas-evolve:profile-brief` skill

**Files:**
- Create: `kernel-evolve/plugins/pallas-evolve/skills/profile-brief/SKILL.md`

**Step 1: Create skill directory**

Run: `mkdir -p kernel-evolve/plugins/pallas-evolve/skills/profile-brief`

**Step 2: Write SKILL.md**

Create `kernel-evolve/plugins/pallas-evolve/skills/profile-brief/SKILL.md` with the following content.

The file has three sections:
1. **Frontmatter + Arguments**: Skill metadata and argument parsing
2. **Procedure**: Steps 1-7 extracted from `start` "Profile Brief Generation" (lines 93-229)
3. **Deep Profiling Signal Reference**: Extracted from `start` (lines 476-485)

```markdown
---
name: profile-brief
description: Use when generating a profile brief from TPU evaluation artifacts — reads eval_result.json, LLO/HLO/trace files, classifies bottleneck, derives optimization priorities, writes profile_brief.md
---

# Generate Profile Brief from TPU Evaluation Artifacts

Read raw profiling artifacts from a TPU evaluation and generate a structured `profile_brief.md` with bottleneck diagnosis, optimization priorities, and LLO/HLO key observations.

## Arguments

Expects arguments in this format:

```
/pallas-evolve:profile-brief <artifacts_dir> [--baseline <baseline_dir>] [--round <N>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `artifacts_dir` | Yes | — | Directory containing `eval_result.json` and optional `llo_final.txt`, `hlo_post_opt.txt`, `trace_events.json` |
| `--baseline` | No | — | Baseline artifacts directory for delta comparison (Round 2+) |
| `--round` | No | `0` | Round number for the profile brief header |

**Parse the arguments** from the invocation. Extract:
- `ARTIFACTS_DIR`: the first positional argument (required)
- `BASELINE_DIR`: value after `--baseline` flag, default empty (no delta table)
- `ROUND_N`: value after `--round` flag, default `0`

If `ARTIFACTS_DIR` is missing, print usage and stop.

## Procedure

### Step 1: Read eval_result.json

Read `{ARTIFACTS_DIR}/eval_result.json`. Extract:
- Top-level: `speedup`, `latency_ms`, `compute_ratio`, `memory_transfer_ratio`
- `metadata.hw_utilization`: all unit utilization percentages, fills/spills
- `metadata.profile`: all deep profiling metrics (VLIW bundles, MXU dual_ratio, HBM bandwidth, arithmetic intensity, compute efficiency, bundle density, DMA analysis, etc.)

If `eval_result.json` does not exist, stop with error: "No eval_result.json found in {ARTIFACTS_DIR}."

### Step 2: Read optional LLO file

If `{ARTIFACTS_DIR}/llo_final.txt` exists, read it and extract key excerpts (max 100 lines total):
- **Inner loop body**: Find the main loop (look for loop markers or repeated MXU op sequences) and extract the body showing VLIW bundle structure
- **MXU scheduling**: Find consecutive `.mxu0`/`.mxu1` operations. Show whether both MXUs are co-scheduled in the same VLIW bundles or separated
- **DMA patterns**: Find `dma.start`/`dma.done` pairs. Show whether DMA overlaps with computation
- **Register spills**: Find `.vmem_store`/`.vmem_load` patterns that indicate register pressure
- **Pipeline bubbles**: Find sequences of `nop` instructions

If the file does not exist, note "LLO not available" for the brief.

### Step 3: Read optional HLO file

If `{ARTIFACTS_DIR}/hlo_post_opt.txt` exists, read it and note:
- Number of `fusion` blocks (0 is ideal for Pallas single-kernel)
- Any `transpose`, `copy`, or `bitcast` operations outside the main custom_call
- Shape information from `tpu_custom_call` parameters

If the file does not exist, note "HLO not available" for the brief.

### Step 4: Classify the bottleneck

Use the multi-signal table:
- `compute_ratio < 0.50` → memory-bound
- `compute_ratio >= 0.75` → compute-bound
- `dual_ratio < 0.5` → single-MXU
- `arithmetic_intensity < 10` → low arithmetic intensity
- `vector_spills > 0` → register pressure
- `vmem_utilization_pct < 30` → VMEM underutilized (room to increase block sizes)
- `vmem_utilization_pct > 90` → VMEM near capacity (OOM risk)
- `hbm_capacity_utilization_pct > 50` → high HBM allocation
- `scalar_alu_util_pct > mxu_util_pct` → scalar-heavy
- Check combined patterns (see analyze skill for full table)

### Step 5: Derive optimization priorities

Rank the top 3 optimization directions that the profile data suggests will have the most impact:
- Memory-bound → prioritize K-tiling, scratch memory, double buffering
- Compute-bound + low dual_ratio → prioritize MXU dual scheduling
- Register pressure → prioritize smaller blocks, fewer intermediates
- VMEM underutilized (<30%) + memory-bound → increase block sizes, add scratch memory to improve on-chip data reuse
- VMEM near capacity (>90%) → do not increase blocks, reduce intermediates
- High HBM allocation (>50%) → eliminate redundant buffers, use in-place updates
- Low ILP (avg_ops_per_bundle < 2) → simplify kernel for better VLIW packing
- High scalar ALU → reduce index computation, simplify control flow

### Step 6: Identify what NOT to try

Directions that won't help given the profile:
- If already compute-bound (compute_ratio > 0.8), don't add more pipelining/prefetch
- If dual_ratio > 0.9, don't focus on MXU utilization
- If no register spills and VMEM usage is low, don't reduce block sizes
- If `vmem_utilization_pct > 90`, don't increase block sizes or add scratch buffers
- If `vmem_utilization_pct < 30` and memory-bound, don't reduce block sizes (VMEM has headroom to grow)

### Step 7: Write profile_brief.md

Write the profile brief to `{ARTIFACTS_DIR}/profile_brief.md` using this template:

````markdown
## Profile Brief for Round {ROUND_N}

### Source
- Kernel: {path to source kernel}
- Speedup: {speedup}x | Latency: {latency_ms}ms
- Compute ratio: {compute_ratio} | Memory transfer ratio: {memory_transfer_ratio}

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | {mxu_util_pct}% | {low(<15)/medium(15-40)/high(>40)} |
| Scalar ALU | {scalar_alu_util_pct}% | {high if > mxu = control-flow heavy} |
| Vector ALU | {vector_alu_util_pct}% | {assessment} |
| Vector Load | {vector_load_util_pct}% | {assessment} |
| Vector Store | {vector_store_util_pct}% | {assessment} |
| Register fills/spills | {fills}/{spills} | {0/0 = ideal, >0 = pressure} |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | {count} | {comparison to baseline if Round 2+} |
| MXU dual ratio | {dual_ratio} | {poor(<0.5)/fair(0.5-0.8)/good(>0.8)} |
| Avg ops/bundle (ILP) | {avg} | {poor(<2)/fair(2-3)/good(>3)} |
| HBM bandwidth | {bytes} | {utilization_pct}% of 3690 GB/s peak |
| Arithmetic intensity | {AI} FLOPs/byte | {low(<10)/medium(10-50)/high(>50)} |
| Compute efficiency | {pct}% | of 2307 TFLOPS peak |
| VMEM utilization | {vmem_pct}% of 64 MiB | {low(<30)/medium(30-70)/good(70-90)/critical(>90)} |
| HBM capacity used | {hbm_cap_pct}% of 192 GB ({peak_memory_mb} MB) | {low/medium/high(>50)} |
| DMA transfers | {count} | {double_buffered: yes/no} |
| Pipeline NOPs | {nop_count} | {low(<10)/medium(10-50)/high(>50)} |

### Bottleneck Diagnosis
**Primary bottleneck**: {memory-bound / compute-bound / register-pressure / scalar-heavy / low-ILP}
**Evidence**: {2-3 sentences citing specific metric values}
**Combined patterns**: {any combined diagnoses from multi-signal analysis}

### LLO Key Observations

**MXU Scheduling** ({mxu0_count} mxu0 ops, {mxu1_count} mxu1 ops, dual_ratio={ratio}):
```
{10-20 lines showing MXU op placement in VLIW bundles}
```

**DMA Pattern** ({dma_count} transfers, {sync_count} syncs):
```
{10-15 lines showing dma.start/dma.done spacing and overlap with compute}
```

**{Additional section if register spills or pipeline bubbles detected}**:
```
{10-15 lines showing the issue}
```

### HLO Key Observations
- Fusions: {count} (0 = ideal for single Pallas kernel)
- {Notable patterns: transposes, copies, shape mismatches}

### Optimization Priorities (derived from profile)
1. **{Direction}**: {Why this is the top priority, citing specific metrics}
2. **{Direction}**: {Why this is second priority}
3. **{Direction}**: {Why this is third priority}

### What NOT to try (profile evidence)
- **{Direction}**: {Why profile shows this won't help, citing metrics}
````

### Step 8: Add delta table (if --baseline provided)

If `BASELINE_DIR` is set, read `{BASELINE_DIR}/eval_result.json` and compare with the current variant's metrics. **Prepend** a delta table section at the top of the profile brief (after the `## Profile Brief` heading):

```markdown
### Delta vs Baseline
| Metric | Baseline | Current Best | Delta |
|--------|----------|-------------|-------|
| Speedup | 1.00x | {x}x | +{pct}% |
| Compute ratio | {base} | {curr} | {+/-} |
| VLIW bundles | {base} | {curr} | {+/-} |
| MXU dual ratio | {base} | {curr} | {+/-} |
| Register spills | {base} | {curr} | {+/-} |
| VMEM utilization | {base}% | {curr}% | {+/-} |
| HBM capacity | {base} MB | {curr} MB | {+/-} |
```

## Deep Profiling Signal Reference

These signals are available in `eval_result.json` under `metadata.profile`:

- `vliw_bundle_count`: Total compiled VLIW bundles. Fewer bundles = simpler kernel = faster. Compare across iterations to detect complexity bloat.
- `mxu_utilization.dual_ratio`: How evenly both MXUs (matrix units) are used. 1.0 = both equally loaded. <0.5 means one MXU is idle — check matmul dimensions.
- `hbm_bandwidth_bytes`: Total HBM memory traffic per invocation. Lower = better. Pallas should keep data in VMEM to avoid HBM round-trips.
- `arithmetic_intensity` (FLOPs/byte): Higher means more compute per byte of memory traffic. Low values indicate memory-bound behavior.
- `compute_efficiency_pct`: Actual throughput vs TPU v7x peak (275 TFLOPS BF16). Shows headroom for optimization.
- `vmem_utilization_pct`: On-chip VMEM usage as % of 64 MiB capacity. Higher is better (more on-chip reuse). <30% = underutilized, >90% = near OOM.
- `hbm_capacity_utilization_pct`: Peak HBM memory usage as % of 192 GB capacity. High values mean large buffer allocations — check for redundant intermediates.

**When analyzing iteration results, check all signals — not just speedup and compute_ratio. VLIW bundle count and MXU dual_ratio are leading indicators of kernel quality.**
```

**Step 3: Verify skill file exists**

Run: `ls -la kernel-evolve/plugins/pallas-evolve/skills/profile-brief/SKILL.md`
Expected: File exists with the content written above

**Step 4: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/profile-brief/SKILL.md
git commit -m "feat(pallas-evolve): create profile-brief skill extracted from start"
```

---

### Task 2: Modify `init-kernel` SKILL.md — replace Step 9d and update Step 10

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md:645-689`

**Step 1: Replace Step 9d**

In `kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md`, replace lines 645-657 (the current optional Step 9d) with the following mandatory baseline profiling step:

Old text (to find and replace):
```
### 9d. TPU verification (optional)

Check if kubectl is available and connected:

```bash
kubectl cluster-info 2>/dev/null
```

If connected, use AskUserQuestion with header "TPU verify" and options:
- "Submit baseline evaluation" — submit a Round 0 evaluation via pallas-evolve:submit to verify correctness on real TPU hardware
- "Skip" — defer to first round of pallas-evolve:start

If the user chooses to submit, use the pallas-evolve:submit skill with the template and ref files.
```

New text:
````
### 9d. Baseline profiling (Round 0)

Verify the template kernel compiles and runs on TPU, collect baseline performance metrics and profiling artifacts.

1. **Verify kubectl connectivity**:

   ```bash
   kubectl cluster-info
   ```

   If not connected, **stop with error**: "TPU connectivity required for baseline profiling. Connect to the GKE cluster (`gcloud container clusters get-credentials tpu7x-cluster --zone us-central1`) and re-run."

2. **Create baseline directory**:

   ```bash
   mkdir -p kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/
   cp kernel-evolve/examples/kernels/${KERNEL_NAME}.py \
      kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/kernel.py
   ```

3. **Set up temporary iteration structure for submit**: The `pallas-evolve:submit` skill expects an iteration directory with `variants/*/kernel.py`. Create a temporary structure:

   ```bash
   BASELINE_TMPDIR=$(mktemp -d)
   mkdir -p "${BASELINE_TMPDIR}/iteration_0/variants/baseline/"
   cp kernel-evolve/examples/kernels/${KERNEL_NAME}.py \
      "${BASELINE_TMPDIR}/iteration_0/variants/baseline/kernel.py"
   ```

4. **Submit baseline for TPU evaluation**: Invoke `pallas-evolve:submit` via the Skill tool, pointing at the temporary iteration directory. This will:
   - Submit the baseline kernel as a single-variant batch
   - Collect `eval_result.json` with performance metrics and deep profiling data
   - Download `llo_final.txt`, `hlo_post_opt.txt`, `trace_events.json` from GCS

5. **Copy results to permanent baseline directory**:

   ```bash
   cp "${BASELINE_TMPDIR}/iteration_0/variants/baseline/"* \
      kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/
   rm -rf "${BASELINE_TMPDIR}"
   ```

6. **Check for compilation failure**: Read `kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/eval_result.json`. If `status` is `COMPILE_ERROR`, **stop and report the error** — the template kernel itself is broken and must be fixed before optimization can begin.

7. **Generate baseline profile brief**: Invoke `pallas-evolve:profile-brief` via the Skill tool:

   ```
   /pallas-evolve:profile-brief kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/ --round 0
   ```

   This writes `profile_brief.md` into the baseline directory.

8. **Commit baseline artifacts**:

   ```bash
   git add kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/
   git commit -m "perf(${KERNEL_NAME}): baseline profiling — Round 0 artifacts"
   ```
````

**Step 2: Update Step 10b summary**

In the same file, find the Step 10b summary block and add the baseline directory to the generated files list.

Old text:
```
Generated files:
  kernel-evolve/examples/{KERNEL_NAME}.yaml              — config
  kernel-evolve/examples/kernels/{KERNEL_NAME}.py         — template (with EVOLVE-BLOCK)
  kernel-evolve/examples/kernels/{KERNEL_NAME}_ref.py     — reference (self-contained)
  kernel-evolve/upstream/{KERNEL_NAME}/                   — unmodified upstream source
  kernel-evolve/tests/test_{KERNEL_NAME}.py               — pytest convention tests
  kernel-evolve/tests/standalone_{KERNEL_NAME}_test.py    — TPU integration test
```

New text:
```
Generated files:
  kernel-evolve/examples/{KERNEL_NAME}.yaml              — config
  kernel-evolve/examples/kernels/{KERNEL_NAME}.py         — template (with EVOLVE-BLOCK)
  kernel-evolve/examples/kernels/{KERNEL_NAME}_ref.py     — reference (self-contained)
  kernel-evolve/examples/kernels/{KERNEL_NAME}_baseline/  — baseline profiling artifacts
  kernel-evolve/upstream/{KERNEL_NAME}/                   — unmodified upstream source
  kernel-evolve/tests/test_{KERNEL_NAME}.py               — pytest convention tests
  kernel-evolve/tests/standalone_{KERNEL_NAME}_test.py    — TPU integration test
```

**Step 3: Verify changes**

Run: `grep -c "Baseline profiling (Round 0)" kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md`
Expected: `1`

Run: `grep -c "TPU verification (optional)" kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md`
Expected: `0`

Run: `grep -c "_baseline/" kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md`
Expected: At least `5` (multiple references in Step 9d + Step 10b)

**Step 4: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md
git commit -m "feat(init-kernel): replace optional TPU verification with mandatory baseline profiling"
```

---

### Task 3: Modify `start` SKILL.md — remove baseline logic, add baseline check, use profile-brief skill

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`

This task has 5 edits to the same file, applied sequentially.

**Step 1: Modify Step 6 — add baseline existence check**

Find the current Step 6:
```
6. **Check kubectl**: Run `kubectl cluster-info` to verify connectivity to the GKE cluster.
```

Replace with:
```
6. **Check kubectl and baseline**: Run `kubectl cluster-info` to verify connectivity to the GKE cluster. Then verify baseline profiling exists:

   ```bash
   ls kernel-evolve/examples/kernels/{kernel_name}_baseline/eval_result.json
   ```

   If the baseline directory or `eval_result.json` is missing, **stop**: "Baseline not found. Run `/pallas-evolve:init-kernel` first to generate baseline profiling."
```

**Step 2: Modify Step 9 — copy baseline into run directory**

Find the current Step 9:
```
9. **Initialize run directory**: Create `kernel-evolve/{output_dir}_{YYYYMMDD_HHMMSS}/`:
   - Copy the config YAML as `config.yaml`
   - Copy the template kernel as `baseline_kernel.py`
   - Initialize `lineages.json`:
     ```json
     {"lineages": [], "pruned": [], "round": 0}
     ```
   - Set `iteration = 0`
```

Replace with:
```
9. **Initialize run directory**: Create `kernel-evolve/{output_dir}_{YYYYMMDD_HHMMSS}/`:
   - Copy the config YAML as `config.yaml`
   - Copy the template kernel as `baseline_kernel.py`
   - Copy baseline artifacts: `cp -r kernel-evolve/examples/kernels/{kernel_name}_baseline/ {run_dir}/baseline/`
   - Initialize `lineages.json`:
     ```json
     {"lineages": [], "pruned": [], "round": 0}
     ```
   - Set `iteration = 0`
```

**Step 3: Remove Step 10 — baseline profiling**

Delete the entire Step 10 block (lines 63-91 in start SKILL.md), which starts with:
```
10. **Baseline profiling (Round 0)**: Submit the unmodified template kernel ...
```
And ends before `## Profile Brief Generation`.

**Step 4: Remove Profile Brief Generation section and deep profiling signals**

Delete the entire "## Profile Brief Generation" section (lines 93-229), which starts with:
```
## Profile Brief Generation
```
And ends before `## Optimization Loop`.

Also delete the "Deep profiling signals" paragraph at the end of the file (lines 476-485), which starts with:
```
**Deep profiling signals (from eval_result.json -> metadata.profile):**
```
And ends with:
```
**When analyzing iteration results, check all signals — not just speedup and compute_ratio. VLIW bundle count and MXU dual_ratio are leading indicators of kernel quality.**
```

**Step 5: Modify Phase 0 — use profile-brief skill**

Find the current Phase 0 section and replace it. Current text:
```
### Phase 0: PROFILE (Generate Profile Brief)

Before generating variants, read the previous round's best variant's raw profiling artifacts and generate a profile brief for sub-agents.

1. **Identify the profile source**:
   - Round 1: Use `{run_dir}/baseline/` (from Step 10)
   - Round 2+: Read `lineages.json`, find the lineage with the highest `best_speedup`, use its `best_kernel` directory (the directory containing the kernel file, which should also contain `eval_result.json`, `llo_final.txt`, etc.)

2. **Check artifact availability**:
   - `eval_result.json` MUST exist (it always does after submit)
   - `llo_final.txt` — if missing, note "LLO not available" in the brief
   - `hlo_post_opt.txt` — if missing, note "HLO not available" in the brief
   - `trace_events.json` — optional, used for compute/sync gap analysis

3. **Generate profile brief**: Follow the **Profile Brief Generation** procedure above. Write the output to `{run_dir}/iteration_{N}/profile_brief.md`.

4. **For Round 2+, generate delta table**: Read `{run_dir}/baseline/eval_result.json` and compare with the current best variant's metrics. Include the delta table in the profile brief.

5. **Read the profile brief into a variable** for passing to sub-agents in Phase 1.
```

Replace with:
```
### Phase 0: PROFILE (Generate Profile Brief)

Before generating variants, generate a profile brief from the previous round's best variant for sub-agents.

1. **Identify the profile source**:
   - Round 1: Use `{run_dir}/baseline/` (copied from `examples/kernels/{kernel_name}_baseline/` during Step 9). The baseline `profile_brief.md` already exists — read it directly and skip to step 4.
   - Round 2+: Read `lineages.json`, find the lineage with the highest `best_speedup`, use its `best_kernel` directory (the directory containing the kernel file, which should also contain `eval_result.json`, `llo_final.txt`, etc.)

2. **Generate profile brief (Round 2+ only)**: Invoke `pallas-evolve:profile-brief` via the Skill tool:

   ```
   /pallas-evolve:profile-brief {profile_source_dir} --baseline {run_dir}/baseline/ --round {N}
   ```

   This generates `profile_brief.md` in the profile source directory with a delta-vs-baseline table.

3. **Copy profile brief to iteration directory**: Copy the generated `profile_brief.md` to `{run_dir}/iteration_{N}/profile_brief.md` for archival.

4. **Read the profile brief into a variable** for passing to sub-agents in Phase 1.
```

**Step 6: Verify changes**

Run: `grep -c "Profile Brief Generation" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`
Expected: `0` (section removed)

Run: `grep -c "Baseline profiling (Round 0)" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`
Expected: `0` (step removed)

Run: `grep -c "deep profiling signals" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`
Expected: `0` (section removed)

Run: `grep -c "pallas-evolve:profile-brief" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`
Expected: `1` (Phase 0 invocation)

Run: `grep -c "_baseline/" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`
Expected: At least `3` (Step 6 check, Step 9 copy, Phase 0 reference)

Run: `grep -c "TPU v7x Pallas Optimization Knowledge" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`
Expected: `1` (section preserved)

**Step 7: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
git commit -m "refactor(start): remove baseline profiling, use profile-brief skill in Phase 0"
```

---

### Task 4: Final verification and squash commit

**Step 1: Verify all three skill files are consistent**

Check cross-references between skills:

```bash
# init-kernel references profile-brief
grep "pallas-evolve:profile-brief" kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md

# start references profile-brief
grep "pallas-evolve:profile-brief" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md

# start no longer has inline Profile Brief procedure
grep -c "## Profile Brief Generation" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md

# start no longer has Step 10 baseline
grep -c "Baseline profiling (Round 0)" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md

# init-kernel has mandatory baseline (not optional)
grep -c "TPU verification (optional)" kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md

# profile-brief skill exists
ls kernel-evolve/plugins/pallas-evolve/skills/profile-brief/SKILL.md
```

Expected: all greps return expected counts (see verification steps in Tasks 1-3), profile-brief SKILL.md exists.

**Step 2: Verify no dangling references**

```bash
# Check start doesn't reference "Profile Brief Generation" procedure
grep -n "Profile Brief Generation" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md

# Check start doesn't reference "Step 10" or "from Step 10"
grep -n "Step 10\|from Step 10" kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
```

Expected: no matches.

**Step 3: Review git log**

Run: `git log --oneline -5`
Expected: 3 new commits (profile-brief creation, init-kernel modification, start modification) on top of existing work.
