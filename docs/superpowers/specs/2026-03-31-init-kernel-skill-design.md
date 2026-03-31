# Design: `pallas-evolve:init-kernel` Skill

**Date**: 2026-03-31
**Status**: Draft

## Overview

A new skill within the `pallas-evolve` plugin that initializes a kernel optimization project from `primatrix/pallas-kernel`. Given a kernel name and branch, it clones the upstream repo, consolidates reference kernel code into self-contained files, generates a template with EVOLVE-BLOCK markers, copies and adapts test cases, generates a YAML config, and verifies baseline correctness.

## Motivation

Currently, porting a kernel from `primatrix/pallas-kernel` for optimization in Glaucis is a manual, multi-step process (as evidenced by the chunk_gla port documented in `docs/plans/`). This skill automates the entire scaffolding process, ensuring:

1. **Consistency**: Every kernel follows the same file structure and naming conventions
2. **Traceability**: Every ref kernel is traceable to its upstream source (repo, branch, commit)
3. **Correctness**: Baseline verification is built into the initialization process
4. **Reproducibility**: The exact upstream state is recorded for future reference

## Arguments

```
/pallas-evolve:init-kernel <kernel_name> [--branch <branch>] [--repo <repo_url>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `kernel_name` | Yes | — | Upstream kernel name, corresponding to `tops/ops/<kernel_name>/` directory |
| `--branch` | No | `main` | Branch in primatrix/pallas-kernel |
| `--repo` | No | `https://github.com/primatrix/pallas-kernel` | Upstream repository URL |

**Example:**
```
/pallas-evolve:init-kernel gla --branch feat/gla-v2
```

## Upstream Repo Structure Assumption

The skill assumes a fixed directory structure in `primatrix/pallas-kernel`:

```
primatrix/pallas-kernel/
├── tops/ops/<kernel_name>/     # Kernel source code
│   ├── __init__.py
│   ├── chunk.py                # Core kernel implementation
│   └── ...                     # Additional modules
└── tests/                      # Test files
    └── test_<kernel_name>*.py  # Or tests/ops/<kernel_name>/
```

## Execution Flow

### Step 1 — Clone upstream

```bash
git clone --depth 1 --branch <branch> <repo_url> /tmp/pallas-kernel-<random>
```

Record the HEAD commit SHA for traceability. Cleanup of this temporary directory happens in Step 10 regardless of success or failure of subsequent steps.

### Step 2 — Discover and validate source files

- Verify `tops/ops/<kernel_name>/` directory exists
- List all `.py` files in the kernel directory
- Search for corresponding test files in `tests/` (patterns: `test_<kernel_name>*.py`, `tests/ops/<kernel_name>/`)
- If kernel directory not found, report error with available kernel directories

### Step 2.5 — Analyze kernel structure

Before copying, analyze the upstream kernel code to understand:

1. **Main compute function**: Identify the primary entry-point function (the function that takes shape parameters and returns outputs). Look for patterns like `def chunk_gla(...)`, `def forward(...)`, or the function called from tests.
2. **`custom_vjp` usage**: Check if the kernel uses `jax.custom_vjp` for forward/backward split. This affects how the `optimized_compute` wrapper is structured.
3. **External dependencies**: Identify any imports outside the standard library and JAX ecosystem (e.g., `qwix`, `tokamax`). If external dependencies are found:
   - Record them for the YAML config (may need a custom K8s job template with pip installs)
   - Warn the user about the dependencies
4. **Function signature**: Determine how the upstream compute function takes its arguments and how to map this to the required `simple_compute(**shape_dict)` / `optimized_compute(**shape_dict)` pattern where evaluate.py passes shape parameters as keyword arguments.
5. **Multi-file structure**: Determine which upstream files contain the kernel logic and how they import each other, since all code must be consolidated into a single self-contained file.

If the upstream code is too complex for automated transformation (e.g., deeply nested external dependencies), generate a scaffold with `TODO` markers for manual completion and inform the user.

### Step 3 — Generate reference kernel

**Critical constraint**: evaluate.py loads kernel files via `exec(kernel_code, exec_globals)`, not Python module imports. All generated files must be fully self-contained — no relative imports, no adjacent subdirectory imports.

Generate `kernel-evolve/examples/kernels/<kernel_name>_ref.py`:

- **Consolidate** all upstream `tops/ops/<kernel_name>/*.py` files into a single self-contained file
- Inline all cross-file imports within the upstream kernel package
- Preserve the algorithm logic exactly — no functional changes
- Export both `simple_compute(**shape_dict)` as the main compute function and `reference_fn(**kwargs)` as a thin wrapper (matching existing convention in `matmul_ref.py`, `chunk_gla_ref.py`)
- Include a private `_make_test_data()` helper for internal use by `simple_compute` (not a public API for evaluate.py — evaluate.py only calls `simple_compute`/`reference_fn`)
- Include source traceability header:
  ```python
  # Source: primatrix/pallas-kernel @ branch: <branch>
  # Commit: <sha>
  # Initialized: <date>
  ```

The upstream source files are preserved unmodified in `kernel-evolve/upstream/<kernel_name>/` for traceability, but this directory is NOT part of the evaluate.py code path.

### Step 4 — Generate template kernel

Generate `kernel-evolve/examples/kernels/<kernel_name>.py`:

- Start from the consolidated ref code
- Place EVOLVE-BLOCK markers **carefully** — NOT around the entire file. The boundary must follow existing conventions:
  - **Outside EVOLVE-BLOCK** (frozen): imports, `_make_test_data()`, constants, `optimized_compute` function signature
  - **Inside EVOLVE-BLOCK** (mutable): Pallas kernel functions, orchestration logic, tiling parameters, block specs, grid configurations, memory layout choices
- Export `optimized_compute(**shape_dict)` as the main compute function
- Include descriptive docstring with:
  - Kernel description and optimization targets
  - Reference dimensions from upstream tests

### Step 5 — Copy test cases

- Copy upstream test files to `kernel-evolve/tests/test_<kernel_name>.py`
- Adapt import paths to match the local repository structure
- Preserve test logic and input shapes

### Step 6 — Generate accuracy comparison test

Create `kernel-evolve/tests/standalone_<kernel_name>_test.py`:

```python
"""Standalone accuracy comparison test for <kernel_name>.

Runs both ref and template on TPU, compares outputs with
np.testing.assert_allclose. Requires TPU hardware.
"""
# Import ref and template compute functions
# Run both with the same shape parameters
# Compare with assert_allclose(ref_output, template_output, atol=..., rtol=...)
```

Input shapes are extracted from upstream test cases.

### Step 7 — Generate YAML config

Create `kernel-evolve/examples/<kernel_name>.yaml`:

```yaml
# Source: primatrix/pallas-kernel @ branch: <branch>
# Commit: <sha>
# Initialized: <date>

kernel:
  name: "<kernel_name>"    # Defaults to kernel_name arg; user may customize
  template: "kernels/<kernel_name>.py"
  reference: "kernels/<kernel_name>_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:
  # Extracted from upstream tests
  - { ... }

correctness:
  method: "allclose"
  rtol: 1e-2
  atol: 1.0      # Conservative default, adjust based on kernel

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

batch:
  variants_per_round: 5
  top_k: 2
  max_active_lineages: 4

session:
  max_iterations: 20
  output_dir: "runs/<kernel_name>"
```

If shapes cannot be automatically extracted from upstream tests, the skill prompts the user to provide them.

**Note**: The `kernel.name` field defaults to the `<kernel_name>` argument but may be customized by the user after generation (e.g., upstream `matmul` might become `tiled_matmul`).

### Step 8 — Validate generated config

Validate the generated YAML against the Pydantic schema:

```python
from kernel_evolve.config import load_config
load_config("kernel-evolve/examples/<kernel_name>.yaml")
```

This catches schema mismatches before the user tries to run `pallas-evolve:start`. If validation fails, fix the YAML and re-validate.

### Step 9 — Verify baseline

Since Pallas TPU kernels require TPU hardware for execution:

- **Syntax & import check**: Verify that both `_ref.py` and template files have valid Python syntax and that all imports resolve correctly
- **TPU verification (optional)**: If the user has TPU access (kubectl connected to GKE cluster), offer to submit a baseline evaluation via `pallas-evolve:submit` to verify correctness on real hardware
- **Note to user**: Full numerical accuracy verification requires TPU hardware and will be performed during the first round of `pallas-evolve:start` (Round 0 baseline)

If syntax/import verification fails, report the error and pause for user intervention.

### Step 10 — Cleanup and summary

- Delete the temporary clone directory (`/tmp/pallas-kernel-<random>`) — **this runs regardless of success or failure of previous steps**
- Output summary of generated files
- Print next-step instruction: `"Run /pallas-evolve:start <kernel_name>.yaml to begin optimization"`

## Output File Structure

After successful initialization:

```
kernel-evolve/examples/
├── <kernel_name>.yaml                    # Generated config
└── kernels/
    ├── <kernel_name>_ref.py              # Self-contained reference: all upstream code inlined, exports simple_compute + reference_fn
    └── <kernel_name>.py                  # Self-contained template: EVOLVE-BLOCK markers around kernel logic, exports optimized_compute

kernel-evolve/upstream/
└── <kernel_name>/                        # Unmodified upstream source (for traceability, not used by evaluate.py)
    ├── __init__.py
    ├── chunk.py
    └── ...

kernel-evolve/tests/
├── test_<kernel_name>.py                 # Copied from upstream (import paths adapted)
└── standalone_<kernel_name>_test.py      # Generated accuracy comparison test (requires TPU)
```

## Integration with Existing System

### evaluate.py Compatibility

evaluate.py loads kernel code via `exec(kernel_code, exec_globals)` and discovers functions by name. The generated files must be fully self-contained (no relative imports) and export:
- **Reference (`_ref.py`)**: `simple_compute(**shape_dict)` as the primary function, plus `reference_fn(**kwargs)` as a thin wrapper
- **Template**: `optimized_compute(**shape_dict)` as the primary function

The private `_make_test_data()` helper is used internally by the compute functions — evaluate.py does NOT call it directly. It calls `simple_compute(**shape)` / `optimized_compute(**shape)` passing shape parameters as keyword arguments.

### pallas-evolve:start Handoff

After init-kernel completes, the generated YAML config is fully compatible with `pallas-evolve:start`:
```
/pallas-evolve:start <kernel_name>.yaml
```

No manual configuration adjustments required.

### Source Traceability

All generated files include source metadata:
```
# Source: primatrix/pallas-kernel @ branch: <branch>
# Commit: <sha>
# Initialized: <date>
```

The unmodified upstream source is preserved in `kernel-evolve/upstream/<kernel_name>/` for reference.

## Skill File Location

```
kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md
```

This places it alongside the existing pallas-evolve skills (start, submit, analyze, reflect).

## Error Handling

| Error | Behavior |
|-------|----------|
| Clone fails (network, auth, branch not found) | Report error, suggest checking branch name and network. Cleanup temp dir. |
| Kernel directory not found in upstream | List available kernel directories, ask user to verify name. Cleanup temp dir. |
| Test files not found in upstream | Warn but continue; generate comparison test from kernel code. |
| Function signature mismatch (no recognizable compute function) | Report available functions, ask user which to use. |
| External dependencies detected | Warn user, note dependency in YAML comments, suggest custom K8s job template. |
| Upstream code too complex for auto-consolidation | Generate scaffold with TODO markers, inform user of manual steps needed. |
| YAML config validation fails | Fix and re-validate; report to user if unfixable. |
| Syntax/import verification fails | Report error details, pause for user review. |
| Files already exist in target location | Ask user: overwrite or skip. |

**Cleanup guarantee**: The temporary clone directory is always deleted, even if an error occurs in any step.

## Non-Goals

- **Ongoing sync with upstream**: This is a one-time init. No automatic update mechanism.
- **Modifying upstream code**: Upstream source files are preserved unmodified in `kernel-evolve/upstream/`.
- **Starting optimization**: The skill ends at initialization. User invokes `pallas-evolve:start` separately.
- **K8s Job template generation**: Uses existing templates; only generates the YAML config.
