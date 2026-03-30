# Profile-Informed THINK Phase Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modify the `pallas-evolve:start` skill to collect baseline profiling data (Round 0), generate a profile brief before each THINK phase, and pass raw LLO/HLO analysis to sub-agents so optimization directions are informed by actual hardware profiling data.

**Architecture:** Add a baseline profiling step (Step 10) after startup that submits the unmodified template kernel via the existing submit skill, downloads LLO/HLO/trace artifacts, and generates a `profile_brief.md`. Add a Phase 0 (PROFILE) before each THINK phase that reads the previous round's best variant's raw profiling artifacts and generates a new profile brief. Modify THINK's sub-agent prompts to include profile brief content and require profile-justified optimization rationale.

**Tech Stack:** Markdown skill file (SKILL.md), bash commands for git operations, existing `pallas-evolve:submit` and `pallas-evolve:analyze` skills.

---

### Task 1: Add Step 10 — Baseline Profiling (Round 0)

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md:55-63` (after step 9, before Optimization Loop)

**Step 1: Read the current file**

Read `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md` to confirm the exact insertion point.

**Step 2: Insert Step 10 after Step 9**

After the line `   - Set `iteration = 0`` (line 62), and before `## Optimization Loop` (line 64), insert:

```markdown
10. **Baseline profiling (Round 0)**: Submit the unmodified template kernel for TPU evaluation to collect profiling artifacts:

    a. Create the baseline evaluation directory:
       ```
       mkdir -p {run_dir}/iteration_0/variants/baseline/
       cp {run_dir}/baseline_kernel.py {run_dir}/iteration_0/variants/baseline/kernel.py
       ```

    b. Invoke `pallas-evolve:submit` via the Skill tool. This will:
       - Submit the baseline kernel as a single-variant batch
       - Collect `eval_result.json` with performance metrics and deep profiling data
       - Download `llo_final.txt`, `hlo_post_opt.txt`, `trace_events.json` from GCS

    c. Copy results to a permanent baseline directory:
       ```
       mkdir -p {run_dir}/baseline/
       cp {run_dir}/iteration_0/variants/baseline/* {run_dir}/baseline/
       ```

    d. **Generate baseline profile brief**: Read the raw profiling artifacts and write `{run_dir}/baseline/profile_brief.md`. See the **Profile Brief Generation** section below for the template and procedure.

    e. **Commit and push baseline artifacts**:
       ```bash
       git add {run_dir}/baseline/
       git commit -m "perf({kernel_name}): baseline profiling — Round 0 artifacts"
       git push
       ```

    f. If the baseline evaluation fails (COMPILE_ERROR), stop and report the error — the template kernel itself is broken and must be fixed before optimization can begin.
```

**Step 3: Verify the edit**

Read the modified file and confirm step 10 is properly placed between step 9 and the Optimization Loop section.

**Step 4: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
git commit -m "feat(pallas-evolve): add Step 10 — baseline profiling (Round 0)"
```

---

### Task 2: Add Profile Brief Generation Section

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md` (insert new section before "## Optimization Loop")

**Step 1: Insert Profile Brief Generation section**

After the new Step 10 and before `## Optimization Loop`, insert a new section:

```markdown
## Profile Brief Generation

Generate a `profile_brief.md` from raw profiling artifacts. This procedure is used both for baseline (Step 10d) and for each round (Phase 0).

**Inputs**: An `eval_result.json` file and optionally `llo_final.txt`, `hlo_post_opt.txt`, `trace_events.json` from the same variant directory.

**Procedure**:

1. Read `eval_result.json`. Extract:
   - Top-level: `speedup`, `latency_ms`, `compute_ratio`, `memory_transfer_ratio`
   - `metadata.hw_utilization`: all unit utilization percentages, fills/spills
   - `metadata.profile`: all deep profiling metrics (VLIW bundles, MXU dual_ratio, HBM bandwidth, arithmetic intensity, compute efficiency, bundle density, DMA analysis, etc.)

2. If `llo_final.txt` exists, read it and extract key excerpts (max 100 lines total):
   - **Inner loop body**: Find the main loop (look for loop markers or repeated MXU op sequences) and extract the body showing VLIW bundle structure
   - **MXU scheduling**: Find consecutive `.mxu0`/`.mxu1` operations. Show whether both MXUs are co-scheduled in the same VLIW bundles or separated
   - **DMA patterns**: Find `dma.start`/`dma.done` pairs. Show whether DMA overlaps with computation
   - **Register spills**: Find `.vmem_store`/`.vmem_load` patterns that indicate register pressure
   - **Pipeline bubbles**: Find sequences of `nop` instructions

3. If `hlo_post_opt.txt` exists, read it and note:
   - Number of `fusion` blocks (0 is ideal for Pallas single-kernel)
   - Any `transpose`, `copy`, or `bitcast` operations outside the main custom_call
   - Shape information from `tpu_custom_call` parameters

4. **Classify the bottleneck** using the multi-signal table from the analyze skill:
   - `compute_ratio < 0.50` → memory-bound
   - `compute_ratio >= 0.75` → compute-bound
   - `dual_ratio < 0.5` → single-MXU
   - `arithmetic_intensity < 10` → low arithmetic intensity
   - `vector_spills > 0` → register pressure
   - `scalar_alu_util_pct > mxu_util_pct` → scalar-heavy
   - Check combined patterns (see analyze skill for full table)

5. **Derive optimization priorities** — rank the top 3 optimization directions that the profile data suggests will have the most impact:
   - Memory-bound → prioritize K-tiling, scratch memory, double buffering
   - Compute-bound + low dual_ratio → prioritize MXU dual scheduling
   - Register pressure → prioritize smaller blocks, fewer intermediates
   - Low ILP (avg_ops_per_bundle < 2) → simplify kernel for better VLIW packing
   - High scalar ALU → reduce index computation, simplify control flow

6. **Identify what NOT to try** — directions that won't help given the profile:
   - If already compute-bound (compute_ratio > 0.8), don't add more pipelining/prefetch
   - If dual_ratio > 0.9, don't focus on MXU utilization
   - If no register spills and VMEM usage is low, don't reduce block sizes

7. Write the profile brief in this template:

   ```markdown
   ## Profile Brief for Round {N}

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
   ```

   For Round 2+, add a delta section at the top:

   ```markdown
   ### Delta vs Baseline
   | Metric | Baseline | Current Best | Delta |
   |--------|----------|-------------|-------|
   | Speedup | 1.00x | {x}x | +{pct}% |
   | Compute ratio | {base} | {curr} | {+/-} |
   | VLIW bundles | {base} | {curr} | {+/-} |
   | MXU dual ratio | {base} | {curr} | {+/-} |
   | Register spills | {base} | {curr} | {+/-} |
   ```
```

**Step 2: Verify the edit**

Read the modified section and confirm it's properly placed.

**Step 3: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
git commit -m "feat(pallas-evolve): add Profile Brief Generation section"
```

---

### Task 3: Add Phase 0 — PROFILE before THINK

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md:64-66` (at the start of Optimization Loop, before Phase 1)

**Step 1: Insert Phase 0 before Phase 1**

After the line `For each round from 1 to `max_iterations`:` and before `### Phase 1: THINK`, insert:

```markdown
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

**Step 2: Verify the edit**

Read the modified section and confirm Phase 0 is properly placed before Phase 1.

**Step 3: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
git commit -m "feat(pallas-evolve): add Phase 0 — PROFILE before THINK"
```

---

### Task 4: Modify Phase 1 THINK — Add Profile Context to Sub-Agents

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md:80-153` (Phase 1 THINK section)

**Step 1: Modify Round 1 sub-agent prompt list**

In the Round 1 section, modify step 2 and the sub-agent prompt list (lines ~85-99). Change step 2 to:

```markdown
2. Prepare shared context for sub-agents: read the template kernel content, AGENT.md content, and **profile brief content** (from Phase 0) into variables
```

Add a new bullet to the sub-agent prompt list after "The TPU v7x hard rules and optimization knowledge from this skill":

```markdown
   - **The profile brief content** (full text of `iteration_1/profile_brief.md` generated in Phase 0), including hardware utilization, bottleneck diagnosis, LLO key observations, and optimization priorities
   - **Direction-specific guidance**: Based on the profile brief's bottleneck diagnosis, explain how this sub-agent's assigned direction relates to the identified bottleneck. For example: "The profile shows compute_ratio=0.35 (memory-bound) with no double buffering. Your direction `hbm_compute_overlap` should focus on adding DMA prefetch to hide the 65% memory transfer time visible in the LLO trace."
```

Modify the "Return a summary" instruction to:

```markdown
     - Return a summary: approach taken, **which profile signal motivated this approach**, expected impact on the identified bottleneck, **which hw utilization metric is expected to improve**, key changes
```

Modify the strategy.md template to include profile justification:

```markdown
   ### Variant: {direction_1}
   **Technical direction**: {direction_name}
   **Profile motivation**: {which profile signal drove this approach}
   **Approach**: {specific optimization technique}
   **Expected impact**: {why this should improve performance, referencing profile metrics}
   **Target metric improvement**: {e.g., "compute_ratio 0.35 → 0.60+", "dual_ratio 0.3 → 0.8+"}
   **Key changes**: {summary of code changes}
```

**Step 2: Modify Round 2+ sub-agent prompt list**

In the Round 2+ section, make the same additions to the sub-agent prompt:

Add after "The TPU v7x hard rules and optimization knowledge from this skill":

```markdown
   - **The profile brief content** (full text of `iteration_{N}/profile_brief.md`), including the delta vs baseline table, hardware utilization, bottleneck diagnosis, and LLO key observations
   - **Direction-specific guidance**: Based on the profile brief's bottleneck diagnosis, explain how this sub-agent's assigned direction relates to the identified bottleneck and what changed since the previous round
```

Modify the return summary instruction and strategy.md template similarly to Round 1.

**Step 3: Verify the edits**

Read the modified Phase 1 section and confirm both Round 1 and Round 2+ include profile context.

**Step 4: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
git commit -m "feat(pallas-evolve): add profile brief context to THINK sub-agent prompts"
```

---

### Task 5: Update COMPACT Phase — Include Profile Brief in State Verification

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md:186-194` (Phase 5 COMPACT section)

**Step 1: Add profile_brief.md to the verification list**

In the Phase 5 COMPACT section, add a new bullet to the state verification checklist:

```markdown
- `profile_brief.md` exists for the current iteration (written by Phase 0)
```

This goes after the existing `strategy.md`, `batch_analysis.md`, `selection.md` line.

**Step 2: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
git commit -m "feat(pallas-evolve): include profile_brief.md in COMPACT state verification"
```

---

### Task 6: Final Review and Validation

**Files:**
- Read: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md` (full file)

**Step 1: Read the complete modified file**

Read the entire SKILL.md file end-to-end and verify:
- Step 10 (baseline profiling) is between step 9 and Optimization Loop
- Profile Brief Generation section is between Step 10 and Optimization Loop
- Phase 0 (PROFILE) is the first phase inside the loop, before Phase 1
- Phase 1 Round 1 sub-agent prompts include profile brief + direction-specific guidance
- Phase 1 Round 2+ sub-agent prompts include profile brief + delta vs baseline
- Phase 5 COMPACT includes profile_brief.md in verification
- No broken markdown formatting (headers, code blocks, lists)
- Strategy.md templates include profile motivation fields
- All references are consistent (paths, variable names)

**Step 2: Verify skill loads correctly**

Run the skill in dry-run by reading it:
```bash
cat kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md | head -5
```
Confirm the frontmatter is intact.

**Step 3: Update the skill description if needed**

If the current skill description doesn't mention profiling, update the frontmatter description to:
```yaml
description: Use when starting a Pallas kernel optimization session on TPU v7x — runs baseline profiling, reads config, creates tracking Issue, runs profile-informed think-submit-analyze-reflect-compact loop with lineage tracking
```

**Step 4: Final commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
git commit -m "refactor(pallas-evolve): update skill description for profile-informed flow"
```
