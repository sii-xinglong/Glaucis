# Glaucis

LLM-driven evolutionary optimizer for JAX Pallas TPU kernels. Automatically discovers high-performance kernel variants through iterative mutation, evaluation on real TPU hardware (GKE TPU v7x), multi-signal profiling, and structured reflection.

## How It Works

Glaucis operates as an interactive Claude Code skill loop:

```
                 pallas-evolve:start
                        |
          +-------------+-------------+
          |                           |
      THINK (LLM)              Read config &
    Generate N kernel           create GitHub
    variant mutations           tracking Issue
          |                           |
          +-------------+-------------+
                        |
                pallas-evolve:submit
            Build payloads, create K8s
            Job on GKE, collect results
                        |
                pallas-evolve:analyze
            Multi-signal bottleneck
            classification, top-K
            lineage selection
                        |
                pallas-evolve:reflect
            Extract failure patterns &
            successful optimizations
            into AGENT.md
                        |
                    compact
                 (loop back)
```

Each iteration:

1. **Think** -- Generate N kernel variants by mutating code between `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` markers, guided by learnings in `AGENT.md`
2. **Submit** -- Package variants into a K8s Job, deploy to GKE TPU v7x, run a 5-stage evaluation pipeline (compile, correctness, performance, XPlane trace, deep IR analysis)
3. **Analyze** -- Classify bottlenecks using compute ratio, VLIW bundles, MXU utilization, arithmetic intensity, HBM bandwidth, register spills, and more. Select top-K lineages for the next round.
4. **Reflect** -- Record failure patterns (`[Fxxx]`) and successful optimizations (`[Sxxx]`) into `AGENT.md`. Post round summary to the GitHub Issue.

## Project Structure

```
Glaucis/
├── kernel-evolve/                    # Core Python package
│   ├── src/kernel_evolve/
│   │   ├── config.py                 # Pydantic YAML config validation
│   │   ├── mutation.py               # EVOLVE-BLOCK extraction/injection
│   │   ├── evaluator.py              # EvalResult/EvalRequest data classes
│   │   ├── kube_evaluator.py         # K8s Job submission via kubectl
│   │   ├── profiler.py               # XPlane trace & IR analysis (624 lines)
│   │   └── docker_evaluate_helpers.py # Batch subprocess dispatch
│   ├── docker/
│   │   ├── Dockerfile                # TPU evaluation container
│   │   └── evaluate.py               # 5-stage evaluator (runs inside K8s Pod)
│   ├── examples/
│   │   ├── matmul.yaml               # Simple tiled matmul config
│   │   ├── chunk_gla.yaml            # Chunked Gated Linear Attention config
│   │   ├── gmm_fp8_blockwise.yaml    # Grouped MatMul FP8 config
│   │   └── kernels/                  # Template & reference implementations
│   ├── plugins/pallas-evolve/        # Claude Code skill plugin
│   │   └── skills/{start,submit,analyze,reflect}/
│   ├── scripts/                      # Utility scripts
│   └── tests/                        # 13 test files
├── .github/
│   ├── workflows/
│   │   ├── kernel-eval.yaml          # TPU evaluation workflow
│   │   └── build-image.yaml          # Docker image build & push
│   └── ci/
│       ├── kernel-eval-job.yaml      # K8s Job template
│       ├── kernel-eval-gmm-job.yaml  # GMM FP8 variant Job template
│       └── xplane-explore-job.yaml   # XPlane exploration Job
├── AGENT.md                          # Accumulated optimization learnings
├── docs/plans/                       # Design & implementation documents
└── LICENSE                           # Apache 2.0
```

## Prerequisites

- Python >= 3.10
- `kubectl` configured for a GKE cluster with TPU v7x nodes
- GCS bucket for profile artifact storage (default: `glaucis-profiles`)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed

## Installation

### 1. Install the Python package

```bash
cd kernel-evolve
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[charts]"    # matplotlib for visualization
pip install -e ".[profile]"   # xprof for trace analysis
```

### 2. Install the pallas-evolve skill plugin

The `pallas-evolve` plugin ships with 4 skills (`start`, `submit`, `analyze`, `reflect`) that drive the optimization loop inside Claude Code.

**Option A: Local plugin (for development)**

Add to your project's `.claude/settings.local.json`:

```json
{
  "enabledPlugins": {
    "pallas-evolve@local": true
  }
}
```

Claude Code will discover the plugin from `kernel-evolve/plugins/pallas-evolve/.claude-plugin/plugin.json`.

**Option B: From repository**

```json
{
  "enabledPlugins": {
    "pallas-evolve@https://github.com/sii-xinglong/Glaucis": true
  }
}
```

### 3. Verify skill availability

Launch Claude Code in the project directory. You should see these skills available:

| Skill | Purpose |
|-------|---------|
| `pallas-evolve:start` | Start an optimization session |
| `pallas-evolve:submit` | Submit a batch of variants for TPU eval |
| `pallas-evolve:analyze` | Analyze batch evaluation results |
| `pallas-evolve:reflect` | Record learnings to AGENT.md |

## Usage

### Start an optimization session

In Claude Code, invoke the start skill with a config:

```
/start examples/matmul.yaml
```

This will:
1. Parse the config and validate shapes/correctness settings
2. Create a GitHub Issue for tracking
3. Enter the think-submit-analyze-reflect loop

### Optimization modes

- **Step-by-step** (default) -- Pauses after each phase for human review
- **Autonomous** -- Runs the full loop unattended until max iterations or target speedup is reached

### Writing a new kernel config

Create a YAML config pointing to your kernel template and reference:

```yaml
kernel:
  name: "my_kernel"
  template: "kernels/my_kernel.py"
  reference: "kernels/my_kernel_ref.py"
  markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:
  - { M: 1024, N: 1024, K: 1024 }

correctness:
  atol: 1e-2
  rtol: 1e-2

evaluator:
  namespace: "default"
  job_template: ".github/ci/kernel-eval-job.yaml"
  repo: "sii-xinglong/Glaucis"
  branch: "main"

tpu:
  cluster: "tpu7x-cluster"
  zone: "us-central1"

session:
  max_iterations: 20
  output_dir: "runs/my_kernel"

batch:
  variants_per_round: 4
  top_k: 2
  max_active_lineages: 6
```

Mark the mutable region in your kernel template:

```python
# EVOLVE-BLOCK-START
def my_kernel(x_ref, y_ref, o_ref):
    # ... kernel code to be optimized ...
# EVOLVE-BLOCK-END
```

### Included kernel targets

| Kernel | Description | Config |
|--------|-------------|--------|
| `matmul` | Tiled matrix multiplication | `examples/matmul.yaml` |
| `chunk_gla` | Chunked Gated Linear Attention (fwd + bwd) | `examples/chunk_gla.yaml` |
| `gmm_fp8_blockwise` | Grouped MatMul with FP8 block-wise quantization | `examples/gmm_fp8_blockwise.yaml` |

## Evaluation Pipeline

The evaluator runs inside a K8s Pod on GKE TPU v7x and executes 5 stages:

1. **Compile** -- `exec()` the kernel, `jax.jit` + lower to HLO
2. **Correctness** -- Compare against reference implementation with `np.testing.assert_allclose`
3. **Performance** -- 10 warmup + 50 timed iterations, report median latency and speedup
4. **XPlane Trace** -- JAX profiler capture, compute ratio, memory transfer ratio, per-unit utilization (MXU, Scalar ALU, Vector ALU, Vector Load/Store)
5. **Deep IR** -- HLO/LLO/Mosaic dump parsing: VLIW bundles, MXU distribution, VMEM allocations, bundle density, DMA analysis, HBM bandwidth, FLOP counts, arithmetic intensity

Profile artifacts (HLO, LLO, trace events) are uploaded to GCS for post-hoc analysis.

## Infrastructure

| Component | Detail |
|-----------|--------|
| Cloud | Google Cloud Platform |
| Cluster | `tpu7x-cluster` (GKE, `us-central1`) |
| TPU | v7x (Ironwood), 2x2x1 topology, 4 chips |
| Peak Compute | 2307 TFLOPS (BF16) |
| HBM Bandwidth | 3690 GB/s |
| Container Registry | `us-central1-docker.pkg.dev/tpu-service-473302/glaucis/kernel-eval` |
| Artifact Storage | GCS bucket `glaucis-profiles` |
| CI Auth | Workload Identity Federation for GitHub Actions |

## Development

```bash
cd kernel-evolve

# Run tests
pytest

# Lint
ruff check src/ tests/
ruff format --check src/ tests/
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).
