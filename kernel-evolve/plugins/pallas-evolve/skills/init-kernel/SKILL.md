---
name: init-kernel
description: Use when initializing a new kernel optimization project from primatrix/pallas-kernel — clones upstream, generates ref/template/test/config files, validates baseline
---

# Initialize Kernel from primatrix/pallas-kernel

Bootstrap a kernel optimization project by importing a kernel from the `primatrix/pallas-kernel` upstream repository. Clones the upstream repo, consolidates multi-file kernel code into self-contained files, generates a template with EVOLVE-BLOCK markers, copies tests, generates a YAML config, and validates.

## Arguments

Expects arguments in this format:

```
/pallas-evolve:init-kernel <kernel_name> [--branch <branch>] [--repo <repo_url>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `kernel_name` | Yes | — | Upstream kernel name, corresponding to `tops/ops/<kernel_name>/` directory |
| `--branch` | No | `main` | Branch in primatrix/pallas-kernel |
| `--repo` | No | `https://github.com/primatrix/pallas-kernel` | Upstream repository URL |

**Parse the arguments** from the invocation. Extract:
- `KERNEL_NAME`: the first positional argument (required)
- `BRANCH`: value after `--branch` flag, default `main`
- `REPO_URL`: value after `--repo` flag, default `https://github.com/primatrix/pallas-kernel`

If `KERNEL_NAME` is missing, print usage and stop.

## Step 1 — Clone upstream repository

Run:

```bash
TMPDIR="/tmp/pallas-kernel-$(date +%s)"
git clone --depth 1 --branch $BRANCH $REPO_URL "$TMPDIR"
```

If the clone fails (network error, branch not found), report the error to the user with suggestions:
- Check that the branch name is correct: `git ls-remote --heads $REPO_URL`
- Check network connectivity
- **Stop here** — do not proceed without a successful clone.

Record the HEAD commit SHA for traceability:

```bash
COMMIT_SHA=$(git -C "$TMPDIR" rev-parse HEAD)
```

Store these values for use in later steps:
- `TMPDIR`: path to the cloned repo
- `COMMIT_SHA`: the exact commit
- `BRANCH`: the branch name
- `REPO_URL`: the repository URL
- `KERNEL_NAME`: the kernel name

**Important**: From this point forward, if ANY step fails, you MUST still execute Step 10 (Cleanup) to delete `$TMPDIR`.

## Step 2 — Discover and validate source files

### 2a. Find kernel source

Check that the kernel directory exists in the cloned repo:

```bash
ls "$TMPDIR/tops/ops/$KERNEL_NAME/"
```

If the directory does not exist:
1. List available kernel directories: `ls "$TMPDIR/tops/ops/"`
2. Report to user: "Kernel '$KERNEL_NAME' not found. Available kernels: [list]. Please check the kernel name."
3. **Go to Step 10** (cleanup) and stop.

Read all `.py` files in the kernel directory. These are the upstream source files that will be consolidated.

### 2b. Find upstream tests

Search for test files:

```bash
find "$TMPDIR/tests" -name "test_${KERNEL_NAME}*" -o -name "test_*${KERNEL_NAME}*" 2>/dev/null
ls "$TMPDIR/tests/ops/$KERNEL_NAME/" 2>/dev/null
```

If no test files found, warn the user: "No upstream test files found for '$KERNEL_NAME'. Will generate comparison test from kernel code, but shapes must be provided manually."

### 2c. Check for existing files

Check if any output files already exist in this repository:

```bash
ls kernel-evolve/examples/kernels/${KERNEL_NAME}_ref.py 2>/dev/null
ls kernel-evolve/examples/kernels/${KERNEL_NAME}.py 2>/dev/null
ls kernel-evolve/examples/${KERNEL_NAME}.yaml 2>/dev/null
```

If any files exist, use AskUserQuestion to ask the user: "Files already exist for '$KERNEL_NAME'. Overwrite existing files or skip?"
- If skip: **Go to Step 10** (cleanup) and stop.
- If overwrite: proceed.

## Step 2.5 — Analyze kernel structure

Read ALL `.py` files in `$TMPDIR/tops/ops/$KERNEL_NAME/`. For each file, analyze:

### 2.5a. Identify the main compute function

Look for the primary entry-point function — the function that takes shape parameters and returns computation results. Search patterns (in priority order):

1. A function called from the upstream test files (most reliable)
2. A function decorated with `@jax.custom_vjp` (indicates a differentiable kernel)
3. A function matching the kernel name (e.g., `def chunk_gla(...)`, `def gla(...)`)
4. A function named `forward`, `compute`, or `main`

Record the function name and its full signature (parameter names and defaults).

### 2.5b. Check for custom_vjp

Search for `@jax.custom_vjp` or `@custom_vjp` in the kernel code. If found, note that the kernel has explicit forward/backward implementations. The `optimized_compute` wrapper must call the `custom_vjp`-decorated function.

### 2.5c. Identify external dependencies

Scan all import statements. Classify each import as:

- **Standard library**: `os`, `sys`, `functools`, etc. — OK
- **JAX ecosystem**: `jax`, `jax.numpy`, `jax.experimental.pallas`, `flax`, etc. — OK
- **External**: anything else (e.g., `qwix`, `tokamax`, `torch`, etc.) — WARN

If external dependencies are found:
1. Warn the user: "External dependencies detected: [list]. These must be available in the evaluation environment."
2. Note them for the YAML config (may need `kernel-eval-gmm-job.yaml` or a custom job template instead of the default `kernel-eval-job.yaml`)

### 2.5d. Map function signature to shape_dict pattern

evaluate.py calls compute functions as `fn(**shape_dict)` where `shape_dict` is a dict like `{"M": 1024, "N": 1024}`. The upstream function must accept these as keyword arguments.

If the upstream function signature is `def compute(M, N, K)` → direct match, use as-is.
If the upstream function signature is `def compute(q, k, v, g)` → the wrapper (`simple_compute`/`optimized_compute`) must create tensors from shape params and call the inner function.

Determine how to write the wrapper function. Reference `kernel-evolve/examples/kernels/chunk_gla_ref.py` and `kernel-evolve/examples/kernels/matmul_ref.py` for examples of both patterns.

### 2.5e. Map multi-file structure

Determine the dependency graph between files in `tops/ops/$KERNEL_NAME/`:
- Which file imports from which other file?
- What is the topological order for inlining?

Plan the consolidation order: inline leaf dependencies first, then files that depend on them, building up to the main entry-point file.

### 2.5f. Complexity gate

If the upstream code meets ANY of these conditions:
- More than 5 source files in `tops/ops/$KERNEL_NAME/`
- External dependencies that cannot be pip-installed
- Deeply nested dependency chains (>3 levels of internal imports)

Then: inform the user that automated consolidation may be incomplete and generate a scaffold with `TODO` markers for sections that need manual review.

## Step 3 — Generate reference kernel

### 3a. Preserve upstream source

Copy the upstream source files unmodified for traceability:

```bash
mkdir -p kernel-evolve/upstream/$KERNEL_NAME
cp -r "$TMPDIR/tops/ops/$KERNEL_NAME/"* kernel-evolve/upstream/$KERNEL_NAME/
```

These files are for reference only — they are NOT used by evaluate.py.

### 3b. Generate self-contained ref file

**Critical**: evaluate.py loads kernel files via `exec(kernel_code, exec_globals)`. The ref file MUST be fully self-contained — no relative imports, no imports from adjacent directories.

Create `kernel-evolve/examples/kernels/${KERNEL_NAME}_ref.py` by consolidating ALL upstream source files into a single file. Follow this structure:

```python
# Source: primatrix/pallas-kernel @ branch: {BRANCH}
# Commit: {COMMIT_SHA}
# Initialized: {TODAY's date YYYY-MM-DD}
"""Reference {KERNEL_NAME} kernel from primatrix/pallas-kernel.

Unmodified algorithm logic consolidated into a single self-contained file.
Used as the correctness baseline for evolutionary optimization.
"""

# --- All imports (from all upstream files, deduplicated) ---
import functools
import jax
import jax.numpy as jnp
# ... (all imports from upstream files, deduplicated)

# --- Inlined code from upstream files (in dependency order) ---
# Code from tops/ops/{KERNEL_NAME}/utils.py (if exists)
# Code from tops/ops/{KERNEL_NAME}/chunk.py
# ... etc.

# --- Private helpers ---

def _make_test_data({shape_params_with_defaults}):
    """Create deterministic test data for the kernel."""
    # Generate input tensors using fixed PRNG seeds
    # Match the upstream test data generation pattern
    ...


# --- Public API for evaluate.py ---

def simple_compute({shape_params_with_defaults}):
    """Run the reference kernel and return a scalar loss for correctness comparison."""
    # Create test data
    # Call the kernel's main compute function
    # Return a scalar (e.g., jnp.sum of the output) for easy diff comparison
    ...


def reference_fn(**kwargs):
    """Generic entry point for evaluate.py function discovery."""
    return simple_compute(**kwargs)
```

Key rules:
1. **Inline all cross-file imports**: Replace `from .utils import foo` with the actual code of `foo` inlined above.
2. **Preserve algorithm logic exactly**: No functional changes to the upstream code.
3. **Deduplicate imports**: Merge import statements from all upstream files.
4. **Use fixed PRNG seeds**: `jax.random.PRNGKey(0)`, `PRNGKey(1)`, etc. for deterministic test data.
5. **Return scalar output**: `simple_compute` should return a scalar (e.g., `jnp.sum(output)`) so evaluate.py can compute `max_diff = max(abs(template_out - ref_out))`.
6. **Match signature convention**: `simple_compute` takes shape params as keyword arguments with defaults. `reference_fn` takes `**kwargs` and delegates.

Reference existing files for the pattern:
- Simple case: Read `kernel-evolve/examples/kernels/matmul_ref.py`
- Complex case: Read `kernel-evolve/examples/kernels/chunk_gla_ref.py`

## Step 4 — Generate template kernel

Create `kernel-evolve/examples/kernels/${KERNEL_NAME}.py` starting from the consolidated ref code. The key difference: add EVOLVE-BLOCK markers around the kernel logic that the evolutionary optimizer will mutate.

### EVOLVE-BLOCK placement rules

The EVOLVE-BLOCK boundary determines what the optimizer can change. **Incorrect placement will break the evaluation pipeline.**

**OUTSIDE the EVOLVE-BLOCK** (frozen — optimizer cannot touch):
- All `import` statements
- `_make_test_data()` function
- Constants and configuration values

**INSIDE the EVOLVE-BLOCK** (mutable — optimizer will modify):
- Pallas kernel functions (`def kernel_body(...)`, etc.)
- `pallas_call` invocations with their `BlockSpec`, `grid`, etc.
- Tiling parameters, block sizes
- Memory layout and transpose strategies
- Loop structures and pipelining
- `optimized_compute` function (signature and body — the optimizer may adjust how it orchestrates the Pallas calls)

### Template file structure

```python
"""{KERNEL_NAME} Pallas TPU kernel — template for evolutionary optimization.

{Description from upstream docstrings}

Optimization targets within the EVOLVE-BLOCK:
  - Kernel fusion strategies
  - Block sizes and tiling
  - Memory layout and transpose strategies
  - Loop structure and pipelining
  - Grid dimensions and BlockSpec configurations
  - Accumulator precision choices

Reference dimensions from upstream:
  {shape info, e.g.: q, k, v: [2, 4096, 16, 128]}
"""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
# ... (all imports, same as ref)


def _make_test_data({shape_params_with_defaults}):
    """Create deterministic test data for the kernel."""
    # MUST be identical to the _make_test_data in the ref file
    ...


# EVOLVE-BLOCK-START
{all Pallas kernel functions}

def optimized_compute({shape_params_with_defaults}):
    """Run the optimized kernel and return a scalar loss."""
    ...
# EVOLVE-BLOCK-END
```

Key rules:
1. **`_make_test_data` MUST be identical** in both ref and template — same PRNG seeds, same shapes, same dtypes.
2. **`optimized_compute` signature MUST match `simple_compute`** — same parameter names and defaults.
3. **The initial template body should be functionally equivalent to the ref** — it will diverge during optimization.
4. **Include Pallas imports** (`pl`, `pltpu`) even if the initial code uses pure JAX — the optimizer will need them.

Reference existing files for the pattern:
- Simple case: Read `kernel-evolve/examples/kernels/matmul.py`
- Complex case: Read `kernel-evolve/examples/kernels/chunk_gla.py`

## Step 5 — Copy upstream test cases

If upstream test files were found in Step 2b:

1. Read each upstream test file
2. Create `kernel-evolve/tests/test_${KERNEL_NAME}.py` with adapted content:
   - Replace upstream import paths (e.g., `from tops.ops.gla import chunk`) with the local exec-based pattern or direct imports
   - Add tests following the existing `test_chunk_gla.py` pattern:

```python
"""Tests for {KERNEL_NAME} kernel conventions and basic correctness."""
import ast
import pathlib

import pytest

from kernel_evolve.config import load_config
from kernel_evolve.mutation import extract_evolve_block

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def test_template_syntax():
    ast.parse((EXAMPLES / "kernels/{KERNEL_NAME}.py").read_text())


def test_reference_syntax():
    ast.parse((EXAMPLES / "kernels/{KERNEL_NAME}_ref.py").read_text())


def test_evolve_block_extractable():
    code = (EXAMPLES / "kernels/{KERNEL_NAME}.py").read_text()
    block = extract_evolve_block(code)
    assert len(block) > 100
    assert "optimized_compute" in block


def test_template_has_optimized_compute():
    ns = {}
    exec((EXAMPLES / "kernels/{KERNEL_NAME}.py").read_text(), ns)
    assert callable(ns.get("optimized_compute"))


def test_reference_has_simple_compute():
    ns = {}
    exec((EXAMPLES / "kernels/{KERNEL_NAME}_ref.py").read_text(), ns)
    assert callable(ns.get("simple_compute"))


def test_reference_has_reference_fn():
    ns = {}
    exec((EXAMPLES / "kernels/{KERNEL_NAME}_ref.py").read_text(), ns)
    assert callable(ns.get("reference_fn"))


def test_matching_test_data():
    """Both template and ref must generate identical test data."""
    tmpl_ns = {}
    ref_ns = {}
    exec((EXAMPLES / "kernels/{KERNEL_NAME}.py").read_text(), tmpl_ns)
    exec((EXAMPLES / "kernels/{KERNEL_NAME}_ref.py").read_text(), ref_ns)
    # Call _make_test_data with small dimensions and compare
    # Use reduced shapes to keep test fast (no TPU needed for data generation)
    {small_shape_args}  # e.g., M=64, N=64
    tmpl_data = tmpl_ns["_make_test_data"]({small_shape_call})
    ref_data = ref_ns["_make_test_data"]({small_shape_call})
    # Compare all returned tensors
    import jax.numpy as jnp
    # If _make_test_data returns a tuple, compare element-wise
    # If it returns a single tensor, compare directly


def test_config_loads():
    config = load_config(EXAMPLES / "{KERNEL_NAME}.yaml")
    assert config.kernel.name == "{KERNEL_NAME}"
    assert len(config.shapes) >= 1
```

3. Preserve any upstream-specific tests that verify kernel-specific behavior (e.g., shape assertions, gradient checks)
4. Extract input shapes from upstream tests for use in Step 7 (YAML config generation)

If no upstream test files were found, generate only the convention tests above using the kernel analysis from Step 2.5.

## Step 6 — Generate standalone accuracy comparison test

Create `kernel-evolve/tests/standalone_${KERNEL_NAME}_test.py` following the pattern in `kernel-evolve/tests/standalone_chunk_gla_test.py`:

```python
"""Standalone integration test for {KERNEL_NAME} kernel.

Run on TPU:
    python tests/standalone_{KERNEL_NAME}_test.py

Verifies:
  1. Both template and reference produce finite results
  2. Template matches reference within atol={ATOL}
  3. Measures forward+backward latency
"""
import time
import sys
import numpy as np

import jax
import jax.numpy as jnp


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Platform: {jax.default_backend()}")

    # Load kernels via exec (same as evaluate.py)
    tmpl_ns = {}
    ref_ns = {}
    exec(open("examples/kernels/{KERNEL_NAME}.py").read(), tmpl_ns)
    exec(open("examples/kernels/{KERNEL_NAME}_ref.py").read(), ref_ns)

    shapes = {SHAPES_DICT}  # e.g., {"M": 1024, "N": 1024, "K": 1024}
    print(f"\nShapes: {shapes}")

    # Correctness
    print("\n--- Correctness ---")
    tmpl_out = tmpl_ns["optimized_compute"](**shapes)
    ref_out = ref_ns["simple_compute"](**shapes)
    jax.block_until_ready(tmpl_out)
    jax.block_until_ready(ref_out)

    max_diff = float(np.max(np.abs(np.array(tmpl_out) - np.array(ref_out))))
    print(f"Template output: {float(tmpl_out):.6f}")
    print(f"Reference output: {float(ref_out):.6f}")
    print(f"Max diff: {max_diff:.6e}")

    if max_diff > {ATOL}:
        print(f"FAIL: max_diff {max_diff} > atol {ATOL}")
        sys.exit(1)
    print("PASS: correctness within tolerance")

    # Performance
    print("\n--- Performance ---")
    warmup = 10
    iters = 50

    for name, fn in [("template", tmpl_ns["optimized_compute"]),
                     ("reference", ref_ns["simple_compute"])]:
        for _ in range(warmup):
            out = fn(**shapes)
            jax.block_until_ready(out)

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn(**shapes)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        print(f"{name}: {median_ms:.2f} ms (median of {iters})")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

Replace `{SHAPES_DICT}` with the primary shape dict extracted from upstream tests (Step 5).
Replace `{ATOL}` with the correctness tolerance (default `1.0`).

## Step 7 — Generate YAML config

Create `kernel-evolve/examples/${KERNEL_NAME}.yaml`:

```yaml
# Source: primatrix/pallas-kernel @ branch: {BRANCH}
# Commit: {COMMIT_SHA}
# Initialized: {TODAY}

kernel:
  name: "{KERNEL_NAME}"
  template: "kernels/{KERNEL_NAME}.py"
  reference: "kernels/{KERNEL_NAME}_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:
{SHAPES_YAML}

correctness:
  method: "allclose"
  rtol: 1e-2
  atol: 1.0

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
  output_dir: "runs/{KERNEL_NAME}"
```

**Shapes**: Extract from upstream tests (Step 5). Format as YAML list of dicts:
```yaml
shapes:
  - { M: 1024, N: 1024, K: 1024 }
  - { M: 2048, N: 2048, K: 2048 }
```

If shapes could not be extracted from upstream tests, use AskUserQuestion to ask the user to provide them.

**External dependencies**: If Step 2.5c detected external dependencies:
- Change `job_template` to a custom template that installs the dependencies
- Add a comment in the YAML noting the dependencies:
  ```yaml
  # External dependencies: qwix, tokamax
  # Using custom job template with pip installs
  evaluator:
    job_template: ".github/ci/kernel-eval-{KERNEL_NAME}-job.yaml"
  ```
- Warn the user that they may need to create the custom job template (reference `.github/ci/kernel-eval-gmm-job.yaml` for the pattern).

## Step 8 — Validate generated config

Validate the generated YAML config against the Pydantic schema:

```bash
cd kernel-evolve && python -c "
from kernel_evolve.config import load_config
config = load_config('examples/${KERNEL_NAME}.yaml')
print(f'Config validated: kernel={config.kernel.name}, shapes={config.shapes}')
print(f'Template: {config.kernel.template}')
print(f'Reference: {config.kernel.reference}')
"
```

If validation fails:
1. Read the error message
2. Fix the YAML file
3. Re-run validation
4. If still failing after 3 attempts, report the error to the user and pause

## Step 9 — Verify baseline

### 9a. Syntax and import verification

Verify both generated kernel files have valid Python syntax:

```bash
cd kernel-evolve && python -c "
import ast
# Check ref
ref_code = open('examples/kernels/${KERNEL_NAME}_ref.py').read()
ast.parse(ref_code)
print('ref: syntax OK')

# Check template
tmpl_code = open('examples/kernels/${KERNEL_NAME}.py').read()
ast.parse(tmpl_code)
print('template: syntax OK')

# Check function discovery (same as evaluate.py)
ref_ns = {}
exec(ref_code, ref_ns)
assert callable(ref_ns.get('simple_compute')), 'ref missing simple_compute'
assert callable(ref_ns.get('reference_fn')), 'ref missing reference_fn'
print('ref: functions OK')

tmpl_ns = {}
exec(tmpl_code, tmpl_ns)
assert callable(tmpl_ns.get('optimized_compute')), 'template missing optimized_compute'
print('template: functions OK')

# Check _make_test_data exists in both
assert callable(ref_ns.get('_make_test_data')), 'ref missing _make_test_data'
assert callable(tmpl_ns.get('_make_test_data')), 'template missing _make_test_data'
print('both: _make_test_data OK')

print('All checks passed.')
"
```

If any check fails, fix the generated file and re-run.

### 9b. Run pytest convention tests

```bash
cd kernel-evolve && python -m pytest tests/test_${KERNEL_NAME}.py -v
```

If tests fail, fix the generated files and re-run.

### 9c. Evolve-block extraction verification

```bash
cd kernel-evolve && python -c "
from kernel_evolve.mutation import extract_evolve_block
code = open('examples/kernels/${KERNEL_NAME}.py').read()
block = extract_evolve_block(code)
print(f'EVOLVE-BLOCK: {len(block)} chars extracted')
assert 'optimized_compute' in block, 'optimized_compute not in evolve block'
print('EVOLVE-BLOCK content OK')
"
```

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

## Step 10 — Cleanup and summary

### 10a. Delete temporary clone

**This step MUST execute regardless of whether previous steps succeeded or failed.**

```bash
rm -rf "$TMPDIR"
```

### 10b. Output summary

Print a summary of all generated files:

```
=== init-kernel complete: {KERNEL_NAME} ===

Source: primatrix/pallas-kernel @ {BRANCH} ({COMMIT_SHA})

Generated files:
  kernel-evolve/examples/{KERNEL_NAME}.yaml              — config
  kernel-evolve/examples/kernels/{KERNEL_NAME}.py         — template (with EVOLVE-BLOCK)
  kernel-evolve/examples/kernels/{KERNEL_NAME}_ref.py     — reference (self-contained)
  kernel-evolve/examples/kernels/{KERNEL_NAME}_baseline/  — baseline profiling artifacts
  kernel-evolve/upstream/{KERNEL_NAME}/                   — unmodified upstream source
  kernel-evolve/tests/test_{KERNEL_NAME}.py               — pytest convention tests
  kernel-evolve/tests/standalone_{KERNEL_NAME}_test.py    — TPU integration test

Next step:
  /pallas-evolve:start {KERNEL_NAME}.yaml
```
