---
name: start
description: Use when starting a Pallas kernel optimization session on TPU v7x — reads config, creates tracking Issue, runs batch think-submit-analyze-reflect-compact loop with lineage tracking
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
   - `diversity_directions`: list of technical directions to guide variant generation (e.g., `tiling_strategy`, `pipeline_depth`, `memory_layout`, `mxu_utilization`, `vectorization`)

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

## Optimization Loop

For each round from 1 to `max_iterations`:

### Phase 1: THINK (Batch Variant Generation)

**CRITICAL RULE**: Each variant MUST explore a genuinely different TECHNICAL DIRECTION. Changing block sizes from 128 to 256 is NOT a different direction — that is a parameter variation. Different directions means fundamentally different optimization APPROACHES:
- Tiling strategy (K-tiling, 2D tiling, diamond tiling)
- Pipeline depth and structure (emit_pipeline, software pipelining, double buffering)
- Memory layout (scratch memory, accumulator placement, data reuse patterns)
- MXU utilization (dual MXU balancing, operand layout for matrix units)
- Vectorization (inner dimension alignment, loop restructuring for SIMD)

#### Round 1 (No lineages yet)

Generate N variants from the baseline kernel, each exploring a DIFFERENT technical direction from the `diversity_directions` list:

1. Re-read AGENT.md failure patterns and successful optimizations
2. For each direction in `diversity_directions` (up to `variants_per_round`):
   - Design an optimization approach specific to that direction
   - Write the mutated kernel to `iteration_1/variants/{direction_name}/kernel.py`
   - Start from the template kernel (keep everything outside EVOLVE-BLOCK unchanged)
   - Replace the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`
   - Preserve function signatures
   - Validate: the file must be valid Python (no syntax errors)
3. Write `iteration_1/strategy.md`:
   ```markdown
   ## Round 1 Strategy

   Generating {N} variants from baseline, each exploring a different technical direction.

   ### Variant: {direction_1}
   **Technical direction**: {direction_name}
   **Approach**: {specific optimization technique}
   **Expected impact**: {why this should improve performance}
   **Key changes**: {summary of code changes}

   ### Variant: {direction_2}
   ...
   ```

#### Round 2+ (Has active lineages)

1. Read `lineages.json` for active lineages
2. Re-read AGENT.md failure patterns and successful optimizations
3. For each active lineage, read its `best_kernel` file
4. Generate variants:
   - If only 1 active lineage: generate N variants from it, each a different direction
   - If multiple active lineages: distribute N variants across lineages (at least 1 per lineage, extras to best-performing lineages)
5. Variant naming convention: `{lineage_id}_{direction}` (e.g., `L1_tiling`, `L2_memory`)
6. Write each variant kernel to `iteration_{N}/variants/{variant_name}/kernel.py`
7. Write `iteration_{N}/strategy.md` covering ALL variants:
   ```markdown
   ## Round {N} Strategy

   Active lineages: {count}
   Total variants this round: {total}

   ### Lineage {L_id} (best speedup: {X}x, direction: {dir})

   #### Variant: {L_id}_{direction_1}
   **Base kernel**: {lineage best_kernel path}
   **Technical direction**: {direction_name}
   **Approach**: {specific optimization technique}
   **Expected impact**: {why this should improve performance}

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
4. **Pipelining**: Use `pltpu.emit_pipeline` to overlap compute and memory transfers across tiles.
5. **Vectorization**: Ensure inner dimensions are multiples of 128 for MXU efficiency.

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

**When analyzing iteration results, check all signals — not just speedup and compute_ratio. VLIW bundle count and MXU dual_ratio are leading indicators of kernel quality.**
