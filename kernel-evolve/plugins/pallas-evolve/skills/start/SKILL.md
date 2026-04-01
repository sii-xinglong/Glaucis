---
name: start
description: Use when starting a Pallas kernel optimization session on TPU v7x — runs baseline profiling, reads config, creates tracking Issue, runs profile-informed think-submit-analyze-reflect-compact loop with lineage tracking
---

# Pallas Kernel Batch Optimization

Optimize a Pallas TPU kernel through batch evolutionary mutation with lineage tracking. Each round generates N variants exploring genuinely different TECHNICAL DIRECTIONS, evaluates them serially in a single GKE Pod, selects top-K as surviving lineages, and iterates. You are the optimization brain.

## Arguments

Expects a config YAML path relative to `kernel-evolve/examples/`:

```
/pallas-evolve:start matmul.yaml
```

## Startup

Execute these steps in order:

1. **Read config**: Read the YAML config file at `kernel-evolve/examples/<arg>`. Extract: kernel name, template path, reference path, shapes, correctness thresholds, evaluator settings (namespace, job_template, repo, branch, poll_interval, timeout), session settings (max_iterations, output_dir).

2. **Read batch config**: From the same YAML, extract the `batch` section:
   - `variants_per_round` (N): how many variants to generate per active lineage per round
   - `top_k` (K): how many best lineages survive each round
   - `max_active_lineages`: cap on total active lineages to prevent exponential growth

   If `batch` section is missing, default to `variants_per_round=1, top_k=1` (single-variant behavior).

3. **Read kernel template**: Read the kernel file at the template path (relative to config dir). Identify the `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` region — this is what you will optimize.

4. **Read reference kernel**: Read the reference implementation. This is the correctness baseline.

5. **Read AGENT.md**: Read `/AGENT.md` at the repo root (if it exists). This contains accumulated learnings:
   - `## Failure Patterns` — known errors and how to avoid them. **You MUST avoid these.**
   - `## Successful Optimizations` — proven strategies. **Prioritize these.**

6. **Check kubectl**: Run `kubectl cluster-info` to verify connectivity to the GKE cluster.

7. **Ask mode**: Use AskUserQuestion with header "Mode" and two options:
   - "Step-by-step" — pause after each round for user review and direction
   - "Autonomous" — run continuously until a termination condition is met

8. **Create GitHub Issue**: Run:
   ```bash
   gh issue create \
     --title "[pallas-evolve] Optimize {kernel_name}" \
     --body "Optimizing {kernel_name} kernel for TPU v7x (batch evolution).\n\nConfig: {config_path}\nShapes: {shapes}\nMax iterations: {max_iterations}\nVariants per round: {variants_per_round}\nTop-K: {top_k}" \
     --label pallas-evolve
   ```
   Save the issue number from the output.

9. **Initialize run directory**: Create `kernel-evolve/{output_dir}_{YYYYMMDD_HHMMSS}/`:
   - Copy the config YAML as `config.yaml`
   - Copy the template kernel as `baseline_kernel.py`
   - Initialize `lineages.json`:
     ```json
     {"lineages": [], "pruned": [], "round": 0}
     ```
   - Set `iteration = 0`

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
   - `vmem_utilization_pct < 30` → VMEM underutilized (room to increase block sizes)
   - `vmem_utilization_pct > 90` → VMEM near capacity (OOM risk)
   - `hbm_capacity_utilization_pct > 50` → high HBM allocation
   - `scalar_alu_util_pct > mxu_util_pct` → scalar-heavy
   - Check combined patterns (see analyze skill for full table)

5. **Derive optimization priorities** — rank the top 3 optimization directions that the profile data suggests will have the most impact:
   - Memory-bound → prioritize K-tiling, scratch memory, double buffering
   - Compute-bound + low dual_ratio → prioritize MXU dual scheduling
   - Register pressure → prioritize smaller blocks, fewer intermediates
   - VMEM underutilized (<30%) + memory-bound → increase block sizes, add scratch memory to improve on-chip data reuse
   - VMEM near capacity (>90%) → do not increase blocks, reduce intermediates
   - High HBM allocation (>50%) → eliminate redundant buffers, use in-place updates
   - Low ILP (avg_ops_per_bundle < 2) → simplify kernel for better VLIW packing
   - High scalar ALU → reduce index computation, simplify control flow

6. **Identify what NOT to try** — directions that won't help given the profile:
   - If already compute-bound (compute_ratio > 0.8), don't add more pipelining/prefetch
   - If dual_ratio > 0.9, don't focus on MXU utilization
   - If no register spills and VMEM usage is low, don't reduce block sizes
   - If `vmem_utilization_pct > 90`, don't increase block sizes or add scratch buffers
   - If `vmem_utilization_pct < 30` and memory-bound, don't reduce block sizes (VMEM has headroom to grow)

7. Write the profile brief using this template:

   ````markdown
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

   For Round 2+, add a delta section at the top of the brief:

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

## Optimization Loop

For each round from 1 to `max_iterations`:

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

### Phase 1: THINK (Batch Variant Generation)

**CRITICAL RULE**: Each variant MUST explore a genuinely different TECHNICAL DIRECTION. Changing block sizes from 128 to 256 is NOT a different direction — that is a parameter variation. Different directions means fundamentally different optimization APPROACHES:
- Tiling strategy (K-tiling, 2D tiling, diamond tiling)
- HBM–compute overlap (double buffering, emit_pipeline, DMA prefetch to hide HBM latency)
- MXU–VPU overlap (schedule matrix ops and vector ops to run concurrently on separate units)
- Memory layout (scratch memory, accumulator placement, data reuse patterns)
- MXU utilization (dual MXU balancing, operand layout for matrix units)
- Vectorization (inner dimension alignment, loop restructuring for SIMD)

**Parallel generation**: Use the Agent tool to generate ALL variants concurrently. Launch one sub-agent per variant in a single message so they run in parallel. Each sub-agent independently writes its kernel file. After all sub-agents complete, the main agent collects their results and writes `strategy.md`.

#### Round 1 (No lineages yet)

Generate N variants from the baseline kernel, each exploring a DIFFERENT technical direction **derived from the profile brief's bottleneck diagnosis and optimization priorities**:

1. Re-read AGENT.md failure patterns and successful optimizations
2. Prepare shared context for sub-agents: read the template kernel content, AGENT.md content, and **profile brief content** (from Phase 0) into variables
3. **Derive N directions from the profile brief**: Read the profile brief's "Optimization Priorities" and "Bottleneck Diagnosis" sections. Based on the diagnosed bottleneck and supporting metrics, select N concrete optimization directions. Each direction must be a genuinely different technical approach motivated by a specific profile signal. For example, if the profile shows `compute_ratio=0.35` (memory-bound) with `dual_ratio=0.0` and `1.9M register spills`, you might derive:
   - `k_tiling` — motivated by low compute_ratio (memory-bound)
   - `register_pressure_reduction` — motivated by 1.9M spills
   - `mxu_dual_scheduling` — motivated by dual_ratio=0.0
   - `dma_prefetch` — motivated by no double buffering in LLO trace
   - `loop_restructuring` — motivated by low ILP (avg_ops_per_bundle)

   **Do NOT use a fixed list of directions.** Each session's directions must be uniquely tailored to what the profile data actually shows.
4. **Dispatch N sub-agents in parallel** (one Agent tool call per direction, all in a single message):
   Each sub-agent receives a prompt containing:
   - The full template kernel content
   - The AGENT.md failure patterns and successful optimizations
   - Its assigned direction name (derived from the profile brief in step 3 above)
   - The TPU v7x hard rules and optimization knowledge from this skill
   - **The profile brief content** (full text of `iteration_1/profile_brief.md` generated in Phase 0), including hardware utilization, bottleneck diagnosis, LLO key observations, and optimization priorities
   - **Direction-specific guidance**: Based on the profile brief's bottleneck diagnosis, explain what specific profile signal motivates this sub-agent's direction and what concrete optimization it should attempt. For example: "The profile shows compute_ratio=0.35 (memory-bound) with no double buffering. Your direction `dma_prefetch` should focus on adding DMA prefetch to hide the 65% memory transfer time visible in the LLO trace."
   - The output path: `iteration_1/variants/{direction_name}/kernel.py`
   - Instructions to:
     - Design an optimization approach specific to its assigned direction
     - Write the mutated kernel file (keep everything outside EVOLVE-BLOCK unchanged)
     - Replace only the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`
     - Preserve function signatures
     - Validate: the file must be valid Python (no syntax errors)
     - Return a summary: approach taken, **which profile signal motivated this approach**, expected impact on the identified bottleneck, **which hw utilization metric is expected to improve**, key changes
5. Collect the returned summaries from all sub-agents and write `iteration_1/strategy.md`:
   ```markdown
   ## Round 1 Strategy

   Generating {N} variants from baseline, each exploring a different technical direction derived from profile analysis.
   Variants generated in parallel via sub-agents.

   ### Variant: {direction_1}
   **Technical direction**: {direction_name}
   **Profile motivation**: {which profile signal drove this approach}
   **Approach**: {specific optimization technique}
   **Expected impact**: {why this should improve performance, referencing profile metrics}
   **Target metric improvement**: {e.g., "compute_ratio 0.35 → 0.60+", "dual_ratio 0.3 → 0.8+"}
   **Key changes**: {summary of code changes}

   ### Variant: {direction_2}
   ...
   ```

#### Round 2+ (Has active lineages)

1. Read `lineages.json` for active lineages
2. Re-read AGENT.md failure patterns and successful optimizations
3. For each active lineage, read its `best_kernel` file
4. **Derive directions from the current round's profile brief**: Read `iteration_{N}/profile_brief.md` (generated in Phase 0). Based on the updated bottleneck diagnosis, delta vs baseline, and optimization priorities, select directions that address the **current** bottleneck — which may differ from previous rounds as the kernel evolves. Avoid repeating directions that the reflect skill (AGENT.md) has flagged as unsuccessful.
5. Plan variant assignments:
   - If only 1 active lineage: assign N different profile-derived directions to it
   - If multiple active lineages: distribute N variants across lineages (at least 1 per lineage, extras to best-performing lineages)
6. Variant naming convention: `{lineage_id}_{direction}` (e.g., `L1_tiling`, `L2_memory`)
7. **Dispatch all variant sub-agents in parallel** (one Agent tool call per variant, all in a single message):
   Each sub-agent receives a prompt containing:
   - The base kernel content (from its assigned lineage's `best_kernel`)
   - The AGENT.md failure patterns and successful optimizations
   - Its assigned direction and lineage context (lineage ID, previous best speedup, prior direction)
   - The TPU v7x hard rules and optimization knowledge from this skill
   - **The profile brief content** (full text of `iteration_{N}/profile_brief.md`), including the delta vs baseline table, hardware utilization, bottleneck diagnosis, and LLO key observations
   - **Direction-specific guidance**: Based on the profile brief's bottleneck diagnosis, explain what specific profile signal motivates this direction and what concrete optimization it should attempt, referencing what changed since the previous round
   - The output path: `iteration_{N}/variants/{variant_name}/kernel.py`
   - Same mutation instructions as Round 1
   - Return a summary: approach taken, **which profile signal motivated this approach**, expected impact on the identified bottleneck, **which hw utilization metric is expected to improve**, key changes
8. Collect the returned summaries from all sub-agents and write `iteration_{N}/strategy.md`:
   ```markdown
   ## Round {N} Strategy

   Active lineages: {count}
   Total variants this round: {total}
   Variants generated in parallel via sub-agents.

   ### Lineage {L_id} (best speedup: {X}x, direction: {dir})

   #### Variant: {L_id}_{direction_1}
   **Base kernel**: {lineage best_kernel path}
   **Technical direction**: {direction_name}
   **Profile motivation**: {which profile signal drove this approach}
   **Approach**: {specific optimization technique}
   **Expected impact**: {why this should improve performance, referencing profile metrics}
   **Target metric improvement**: {e.g., "compute_ratio 0.35 → 0.60+", "dual_ratio 0.3 → 0.8+"}

   #### Variant: {L_id}_{direction_2}
   ...
   ```

### Phase 2: SUBMIT

Invoke `pallas-evolve:submit` via the Skill tool.

The submit skill will:
- Collect all variant kernels into a batch payload
- Create a single K8s Job that evaluates all variants serially via subprocess isolation
- Parse multiple `EVAL_RESULT:` lines from the Job logs
- Save individual results to `iteration_{N}/variants/{variant_name}/eval_result.json`

### Phase 3: ANALYZE

Invoke `pallas-evolve:analyze` via the Skill tool.

The analyze skill will:
- Read all variant results for the round
- Rank by speedup, compare across directions
- Select top-K lineages (create new lineages or update existing ones)
- Enforce `max_active_lineages` cap
- Update `lineages.json` with new state
- Write `iteration_{N}/batch_analysis.md` and `iteration_{N}/selection.md`

### Phase 4: REFLECT

Invoke `pallas-evolve:reflect` via the Skill tool.

The reflect skill will:
- Extract learnings from ALL variant results (batch)
- Update AGENT.md with failure patterns and successful optimizations
- Post a summary comment on the GitHub Issue with a table of all variant results

### Phase 5: COMPACT

Before compacting, verify all critical state is persisted to files:
- `lineages.json` is up to date (written by analyze)
- All `eval_result.json` files exist for each variant
- `strategy.md`, `batch_analysis.md`, `selection.md` are written
- `profile_brief.md` exists for the current iteration (written by Phase 0)
- AGENT.md is updated (written by reflect)

Then invoke `/compact` to compress conversation context. After compaction, the loop will reconstruct state from files.

### Phase 6: CONTINUE?

Read the updated `lineages.json` (written by the analyze skill).

**Step-by-step mode**: Present a lineage summary table and ask the user:

```
## Round {N} Summary

| Lineage | Best Speedup | Direction | Rounds Active | Last Improvement |
|---------|-------------|-----------|---------------|------------------|
| L1      | 1.62x       | tiling    | 3             | Round 3          |
| L2      | 1.45x       | memory    | 2             | Round 2          |

Total variants evaluated this round: {count}
Best overall speedup: {best}x (Lineage {id})
```

Options: "Continue", "Adjust strategy", "Stop"

If "Adjust strategy": ask what directions to prioritize, which lineages to prune, or whether to change variants_per_round.

**Autonomous mode**: Check termination conditions:
- `round >= max_iterations` -> stop
- ALL active lineages have not improved for 3 consecutive rounds -> stop (stagnation)
- Otherwise -> continue to next round

## End of Session

When the loop terminates:

1. **Report final results**: For each lineage, report its best speedup, direction, and evolution history. Highlight the overall best lineage.

2. **Create PR from best lineage**: If the best lineage's speedup > 1.0:
   - Identify the best lineage from `lineages.json` (highest `best_speedup`)
   - Read its `best_kernel` file
   - Create a branch: `git checkout -b pallas-evolve/{kernel_name}-{timestamp}`
   - Copy the best lineage's kernel to the template kernel path
   - Commit and push
   - Create PR:
     ```bash
     gh pr create \
       --title "[pallas-evolve] {kernel_name}: 1.0x -> {best_speedup}x" \
       --body "Optimized via pallas-evolve batch evolution.\n\nBest lineage: {lineage_id} ({direction})\nRounds: {total_rounds}\nVariants evaluated: {total_variants}\n\nSee #{issue_number} for iteration history."
     ```

3. **Close the GitHub Issue** with a final summary comment including:
   - Best speedup achieved and which lineage/direction
   - Total rounds and variants evaluated
   - Final lineage table with all active and pruned lineages

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
4. **HBM–compute overlap**: Use `pltpu.emit_pipeline` or manual double buffering to overlap HBM DMA transfers with MXU/VPU compute. Prefetch the next tile while computing the current one.
5. **MXU–VPU overlap**: Schedule MXU matmul operations and VPU element-wise/reduction operations to execute concurrently. TPU can run both units in parallel when data dependencies allow.
6. **Vectorization**: Ensure inner dimensions are multiples of 128 for MXU efficiency.

**Common pitfalls:**
- Block size 512 may OOM on 2048x2048 matrices (VMEM limit)
- Accumulator dtype should be `jnp.float32` for numerical stability, cast to `bfloat16` on store
- Grid dimensions must evenly divide matrix dimensions

**Deep profiling signals (from eval_result.json -> metadata.profile):**
- `vliw_bundle_count`: Total compiled VLIW bundles. Fewer bundles = simpler kernel = faster. Compare across iterations to detect complexity bloat.
- `mxu_utilization.dual_ratio`: How evenly both MXUs (matrix units) are used. 1.0 = both equally loaded. <0.5 means one MXU is idle — check matmul dimensions.
- `hbm_bandwidth_bytes`: Total HBM memory traffic per invocation. Lower = better. Pallas should keep data in VMEM to avoid HBM round-trips.
- `arithmetic_intensity` (FLOPs/byte): Higher means more compute per byte of memory traffic. Low values indicate memory-bound behavior.
- `compute_efficiency_pct`: Actual throughput vs TPU v7x peak (275 TFLOPS BF16). Shows headroom for optimization.
- `vmem_utilization_pct`: On-chip VMEM usage as % of 64 MiB capacity. Higher is better (more on-chip reuse). <30% = underutilized, >90% = near OOM.
- `hbm_capacity_utilization_pct`: Peak HBM memory usage as % of 192 GB capacity. High values mean large buffer allocations — check for redundant intermediates.

**When analyzing iteration results, check all signals — not just speedup and compute_ratio. VLIW bundle count and MXU dual_ratio are leading indicators of kernel quality.**
