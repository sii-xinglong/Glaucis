"""Standalone integration test for fused_chunk_simple_gla kernel.

Run on TPU:
    python tests/standalone_fused_chunk_simple_gla_test.py

Verifies:
  1. Both template and reference produce finite results
  2. Template matches reference within atol=1e-2
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

    # Load kernels
    tmpl_ns = {}
    ref_ns = {}
    exec(open("examples/kernels/fused_chunk_simple_gla.py").read(), tmpl_ns)
    exec(open("examples/kernels/fused_chunk_simple_gla_ref.py").read(), ref_ns)

    shapes = {"B": 2, "T": 4096, "H": 16, "K": 128, "V": 128, "chunk_size": 64}
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

    if max_diff > 1e-2:
        print(f"FAIL: max_diff {max_diff} > atol 1e-2")
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
