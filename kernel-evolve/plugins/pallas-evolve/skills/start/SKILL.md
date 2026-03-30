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

**Deep profiling signals (from eval_result.json → metadata.profile):**
- `vliw_bundle_count`: Total compiled VLIW bundles. Fewer bundles = simpler kernel = faster. Compare across iterations to detect complexity bloat.
- `mxu_utilization.dual_ratio`: How evenly both MXUs (matrix units) are used. 1.0 = both equally loaded. <0.5 means one MXU is idle — check matmul dimensions.
- `hbm_bandwidth_bytes`: Total HBM memory traffic per invocation. Lower = better. Pallas should keep data in VMEM to avoid HBM round-trips.
- `arithmetic_intensity` (FLOPs/byte): Higher means more compute per byte of memory traffic. Low values indicate memory-bound behavior.
- `compute_efficiency_pct`: Actual throughput vs TPU v7x peak (275 TFLOPS BF16). Shows headroom for optimization.

**When analyzing iteration results, check all signals — not just speedup and compute_ratio. VLIW bundle count and MXU dual_ratio are leading indicators of kernel quality.**
