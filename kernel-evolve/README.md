# kernel-evolve

Evolutionary TPU kernel optimizer. Uses LLM-driven mutation and MAP-Elites search to automatically discover high-performance JAX Pallas kernel variants.

## How It Works

```
                     kernel-evolve run --config matmul.yaml
                                    |
                          Evolution Engine
                         (MAP-Elites loop)
                           /            \
                    LLM Mutation      Population
                     (Claude /       (MAP-Elites
                    Gemini /         archive +
                     Codex)          islands)
                          \           /
                    Evaluation Pipeline
                  [Compile -> Correct -> Perf]
                              |
                    GitHub Actions CI
                  (TPU pod on GKE cluster)
```

### Core Loop

Each generation, for each island:

1. **Select** -- Tournament-select a parent kernel from the MAP-Elites archive (or use the template on generation 1)
2. **Extract** -- Pull the mutable code between `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` markers
3. **Mutate** -- Send the code block, fitness history, failed attempts, and an optimization focus hint to an LLM; receive a mutated code block
4. **Validate** -- Check the mutated kernel has valid Python syntax locally (fail fast, no TPU time wasted)
5. **Evaluate** -- Dispatch to a remote TPU via GitHub Actions CI workflow:
   - **Stage 1: Compilation** -- `jax.jit` + lower to HLO
   - **Stage 2: Correctness** -- Run against reference implementation with `np.testing.assert_allclose`
   - **Stage 3: Performance** -- 10 warmup + 50 timed iterations, report speedup
6. **Insert** -- If the variant is fitter than the current archive occupant at its behavioral descriptor cell, replace it

### MAP-Elites Population

The archive is a grid keyed by three behavioral descriptors, ensuring structural diversity:

| Descriptor | Values | Purpose |
|---|---|---|
| `block_size` | 64, 128, 256, 512 | Tiling granularity |
| `pipeline_stages` | 1, 2, 3, 4 | Memory/compute overlap |
| `memory_strategy` | scratch, hbm, rmw | VMEM usage pattern |

### Island Model

Multiple islands evolve independently with separate archives. Top variants migrate between islands periodically to share good ideas without losing diversity.

### Stagnation Response

When fitness plateaus for `stagnation_limit` generations, the engine rotates the LLM's optimization focus through: tiling -> memory access -> compute optimization -> vectorization -> pipelining.

## Quick Start

```bash
pip install -e ".[dev,anthropic]"

# Validate config
kernel-evolve run --config examples/matmul.yaml --dry-run

# Run evolution
kernel-evolve run --config examples/matmul.yaml

# Check progress
kernel-evolve status runs/matmul_001

# Extract best kernel
kernel-evolve best runs/matmul_001
```

## Configuration

See [`examples/matmul.yaml`](examples/matmul.yaml) for a full example. Key sections:

```yaml
kernel:
  name: "tiled_matmul"
  template: "kernels/matmul.py"       # Pallas kernel with EVOLVE-BLOCK markers
  reference: "kernels/matmul_ref.py"  # Reference implementation for correctness

shapes:                               # Test shapes for evaluation
  - { M: 1024, N: 1024, K: 1024 }

evolution:
  num_islands: 3                      # Parallel evolutionary islands
  max_generations: 50
  stagnation_limit: 10

llm:
  provider: "anthropic"               # anthropic | google | openai
  model: "claude-sonnet-4-6"
  temperature: 0.7
```

## CI Evaluation

Kernel evaluation runs on TPU via GitHub Actions. The engine:
1. Triggers a `workflow_dispatch` event with the kernel code (base64-encoded)
2. Polls for completion via `gh run view`
3. Parses `EVAL_RESULT:{json}` from CI logs

The workflow template is at [`.github/workflows/kernel-eval.yaml`](.github/workflows/kernel-eval.yaml). The evaluation container is at [`docker/`](docker/).

## Output

```
runs/matmul_001/
├── perf_log.md              # Per-generation performance table
├── population/
│   ├── archive_island_*.json  # MAP-Elites archive checkpoints
│   └── variants/              # All generated kernel files
├── best/
│   └── kernel.py              # Best kernel found
└── failed/                    # Failed mutation logs
```

## Project Structure

```
kernel-evolve/
├── src/kernel_evolve/
│   ├── cli.py              # Click CLI (run, status, best)
│   ├── config.py           # Pydantic YAML config validation
│   ├── engine.py           # MAP-Elites evolution loop
│   ├── population.py       # Archive, Variant, BehaviorDescriptor
│   ├── mutation.py         # EVOLVE-BLOCK extraction/injection, syntax check
│   ├── evaluator.py        # EvalRequest/EvalResult, three-stage pipeline
│   ├── ci_dispatcher.py    # GitHub Actions workflow dispatch + log parsing
│   ├── perf_log.py         # Markdown performance log writer
│   └── llm/
│       ├── base.py         # LLMProvider abstract interface
│       ├── anthropic_provider.py
│       ├── google_provider.py
│       └── openai_provider.py
├── docker/
│   ├── Dockerfile          # TPU evaluation container
│   └── evaluate.py         # Three-stage evaluator script
├── examples/
│   ├── matmul.yaml
│   └── kernels/
│       ├── matmul.py       # Template with EVOLVE-BLOCK markers
│       └── matmul_ref.py   # Reference implementation
└── tests/                  # 41 tests, pytest
```

## LLM Providers

Install the provider you need:

```bash
pip install -e ".[anthropic]"   # Claude
pip install -e ".[google]"      # Gemini
pip install -e ".[openai]"      # Codex / GPT
```

Set the corresponding API key environment variable (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`).
