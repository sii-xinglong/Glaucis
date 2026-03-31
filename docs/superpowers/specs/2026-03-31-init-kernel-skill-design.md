# Design: `pallas-evolve:init-kernel` Skill

**Date**: 2026-03-31
**Status**: Draft

## Overview

A new skill within the `pallas-evolve` plugin that initializes a kernel optimization project from `primatrix/pallas-kernel`. Given a kernel name and branch, it clones the upstream repo, copies the reference kernel code, generates a template with EVOLVE-BLOCK markers, copies and adapts test cases, generates a YAML config, and verifies baseline correctness.

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

Record the HEAD commit SHA for traceability.

### Step 2 — Discover and validate source files

- Verify `tops/ops/<kernel_name>/` directory exists
- List all `.py` files in the kernel directory
- Search for corresponding test files in `tests/` (patterns: `test_<kernel_name>*.py`, `tests/ops/<kernel_name>/`)
- If kernel directory not found, report error with available kernel directories

### Step 3 — Copy reference kernel

- Copy the entire `tops/ops/<kernel_name>/` directory to `kernel-evolve/examples/kernels/<kernel_name>/`
- Preserve all original filenames and structure — **no modifications** to the upstream code
- Create entry file `kernel-evolve/examples/kernels/<kernel_name>_ref.py` that:
  - Imports from the copied `<kernel_name>/` subdirectory
  - Re-exports the compute function as `reference_fn` or `simple_compute` (matching evaluate.py's discovery convention)
  - Exports `_make_test_data()` for test input generation
  - Includes source traceability comment:
    ```python
    # Source: primatrix/pallas-kernel @ branch: <branch>
    # Commit: <sha>
    # Initialized: <date>
    ```

### Step 4 — Generate template kernel

- Copy the kernel code and consolidate into `kernel-evolve/examples/kernels/<kernel_name>.py`
- Wrap the entire file content between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers
- Export `optimized_compute` function (alias to the upstream compute function if names differ)
- Export `_make_test_data()` for consistency with evaluate.py
- Include descriptive docstring with:
  - Kernel description
  - Optimization targets
  - Reference dimensions from upstream tests

### Step 5 — Copy test cases

- Copy upstream test files to `kernel-evolve/tests/test_<kernel_name>.py`
- Adapt import paths to match the local repository structure
- Preserve test logic and input shapes

### Step 6 — Generate accuracy comparison test

Create `kernel-evolve/tests/standalone_<kernel_name>_test.py`:

```python
"""Standalone accuracy comparison test for <kernel_name>.

Runs both ref (from primatrix/pallas-kernel) and template,
compares outputs with np.testing.assert_allclose.
"""
# Import ref and template compute functions
# Generate test data using _make_test_data()
# Run both, compare with assert_allclose(ref_output, template_output, atol=..., rtol=...)
```

Input shapes are extracted from upstream test cases.

### Step 7 — Generate YAML config

Create `kernel-evolve/examples/<kernel_name>.yaml`:

```yaml
# Source: primatrix/pallas-kernel @ branch: <branch>
# Commit: <sha>
# Initialized: <date>

kernel:
  name: "<kernel_name>"
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

### Step 8 — Verify baseline

- Run the standalone accuracy comparison test locally (if JAX is available)
- Verify that ref and template produce identical outputs
- If verification fails, report the discrepancy and pause for user intervention

### Step 9 — Cleanup

- Delete the temporary clone directory (`/tmp/pallas-kernel-<random>`)
- Output summary of generated files
- Print next-step instruction: `"Run /pallas-evolve:start <kernel_name>.yaml to begin optimization"`

## Output File Structure

After successful initialization:

```
kernel-evolve/examples/
├── <kernel_name>.yaml                    # Generated config
└── kernels/
    ├── <kernel_name>/                    # Upstream source (unmodified copy)
    │   ├── __init__.py
    │   ├── chunk.py
    │   └── ...
    ├── <kernel_name>_ref.py              # Reference entry: imports from <kernel_name>/, exports reference_fn
    └── <kernel_name>.py                  # Template: consolidated code + EVOLVE-BLOCK markers, exports optimized_compute

kernel-evolve/tests/
├── test_<kernel_name>.py                 # Copied from upstream (import paths adapted)
└── standalone_<kernel_name>_test.py      # Generated accuracy comparison test
```

## Integration with Existing System

### evaluate.py Compatibility

The generated files follow evaluate.py's function discovery conventions:
- **Reference**: `reference_fn` or `simple_compute` in `_ref.py`
- **Template**: `optimized_compute` or `kernel_fn` in the template file
- **Test data**: `_make_test_data()` in both files

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

This enables future comparison with upstream updates.

## Skill File Location

```
kernel-evolve/plugins/pallas-evolve/skills/init-kernel/SKILL.md
```

This places it alongside the existing pallas-evolve skills (start, submit, analyze, reflect).

## Error Handling

| Error | Behavior |
|-------|----------|
| Clone fails (network, auth, branch not found) | Report error, suggest checking branch name and network |
| Kernel directory not found in upstream | List available kernel directories, ask user to verify name |
| Test files not found in upstream | Warn but continue; generate comparison test from kernel code |
| Function signature mismatch (no recognizable compute function) | Report available functions, ask user which to use |
| Baseline verification fails (ref != template) | Report numerical diff, pause for user review |
| Files already exist in target location | Ask user: overwrite or skip |

## Non-Goals

- **Ongoing sync with upstream**: This is a one-time init. No automatic update mechanism.
- **Modifying upstream code**: Ref files are unmodified copies.
- **Starting optimization**: The skill ends at initialization. User invokes `pallas-evolve:start` separately.
- **K8s Job template generation**: Uses existing templates; only generates the YAML config.
