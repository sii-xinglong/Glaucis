"""Explore xplane.pb tool types on TPU to discover available hardware unit data.

Runs a simple matmul Pallas kernel, captures an xplane trace, then dumps
raw output from overview_page, op_profile, hlo_stats, framework_op_stats,
and roofline_model tool types.

Usage: Run on a TPU node with xprof installed.
  python kernel-evolve/scripts/explore_xplane.py
"""

import json
import os
import sys
import traceback
from pathlib import Path


def run_matmul_kernel():
    """Run a simple Pallas matmul and return the kernel function."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    M, N, K = 1024, 1024, 1024
    BLOCK_M, BLOCK_N = 128, 128

    def matmul_kernel(x_ref, y_ref, o_ref):
        acc = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype)

    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=jnp.bfloat16)

    def compute():
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
            grid=(M // BLOCK_M, N // BLOCK_N),
            in_specs=[
                pl.BlockSpec((BLOCK_M, K), lambda i, j: (i, 0)),
                pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
        )(x, y)

    return compute


def capture_trace(compute_fn, trace_dir="/tmp/xplane_explore"):
    """Capture xplane trace from kernel execution."""
    import jax

    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    # Warmup
    for _ in range(5):
        out = compute_fn()
        out.block_until_ready()

    # Profile
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

    jax.profiler.start_trace(trace_dir, profiler_options=options)
    for _ in range(5):
        out = compute_fn()
        out.block_until_ready()
    jax.profiler.stop_trace()

    # Find xplane.pb
    for root, _dirs, files in os.walk(trace_dir):
        for f in files:
            if f.endswith(".xplane.pb"):
                return os.path.join(root, f)
    return None


def dump_tool(xplane_path, tool_name, params=None):
    """Call xspace_to_tool_data and return raw result."""
    from xprof.convert import raw_to_tool_data

    if params is None:
        params = {}
    try:
        data, content_type = raw_to_tool_data.xspace_to_tool_data(
            [xplane_path], tool_name, params
        )
        return {
            "ok": True,
            "content_type": content_type,
            "data_type": type(data).__name__,
            "data_length": len(data) if data else 0,
            "data": data,
        }
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}


def safe_json_parse(raw_data):
    """Try to parse as JSON, return parsed or raw string."""
    if isinstance(raw_data, bytes):
        raw_data = raw_data.decode("utf-8", errors="replace")
    try:
        return json.loads(raw_data)
    except (json.JSONDecodeError, TypeError):
        return raw_data


def print_section(title, content, max_lines=200):
    """Print a section with truncation for large outputs."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    if isinstance(content, dict) and content.get("ok") is False:
        print(f"ERROR: {content['error']}")
        return

    if isinstance(content, dict) and content.get("ok"):
        print(f"Content-Type: {content['content_type']}")
        print(f"Data type: {content['data_type']}, Length: {content['data_length']}")
        raw = content["data"]
        parsed = safe_json_parse(raw)
        if isinstance(parsed, dict):
            formatted = json.dumps(parsed, indent=2, default=str)
        elif isinstance(parsed, list):
            formatted = json.dumps(parsed, indent=2, default=str)
        else:
            formatted = str(parsed)
    else:
        formatted = json.dumps(content, indent=2, default=str)

    lines = formatted.split("\n")
    if len(lines) > max_lines:
        for line in lines[:max_lines]:
            print(line)
        print(f"\n... TRUNCATED ({len(lines) - max_lines} more lines) ...")
    else:
        print(formatted)


def main():
    print("=" * 80)
    print("  XPLANE TOOL TYPE EXPLORATION")
    print("=" * 80)

    # Step 1: Check environment
    import jax
    print(f"\nJAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Platform: {jax.default_backend()}")

    # Step 2: Run kernel and capture trace
    print("\n--- Running matmul kernel ---")
    compute_fn = run_matmul_kernel()
    print("Kernel compiled. Capturing trace...")

    xplane_path = capture_trace(compute_fn)
    if xplane_path is None:
        print("ERROR: No .xplane.pb file generated")
        sys.exit(1)
    print(f"Trace captured: {xplane_path}")
    print(f"File size: {os.path.getsize(xplane_path)} bytes")

    # Step 3: Query available tools for this xplane
    print("\n--- Querying available tool names ---")
    try:
        from xprof.convert import raw_to_tool_data
        tool_names_data, _ = raw_to_tool_data.xspace_to_tool_data(
            [xplane_path], "tool_names", {}
        )
        tool_names = safe_json_parse(tool_names_data)
        print(f"Available tools: {json.dumps(tool_names, indent=2)}")
    except Exception:
        print(f"tool_names query failed: {traceback.format_exc()}")
        tool_names = []

    # Step 4: Dump each tool type
    tools_to_explore = [
        ("overview_page", {}),
        ("op_profile", {}),
        ("hlo_stats", {}),
        ("framework_op_stats", {}),
        ("roofline_model", {}),
    ]

    for tool_name, params in tools_to_explore:
        print(f"\n--- Dumping: {tool_name} ---")
        result = dump_tool(xplane_path, tool_name, params)
        print_section(tool_name, result, max_lines=300)

    # Step 5: Also dump trace_viewer event names for reference
    print("\n--- Trace viewer: unique event names by category ---")
    tv_result = dump_tool(xplane_path, "trace_viewer", {})
    if tv_result.get("ok"):
        events = safe_json_parse(tv_result["data"])
        if isinstance(events, dict):
            events = events.get("traceEvents", [])

        # Categorize events
        categories = {}
        for evt in events:
            cat = evt.get("cat", "unknown")
            name = evt.get("name", "")
            if cat not in categories:
                categories[cat] = set()
            categories[cat].add(name)

        for cat in sorted(categories.keys()):
            names = sorted(categories[cat])
            print(f"\n  Category '{cat}': {len(names)} unique names")
            for n in names[:20]:
                print(f"    - {n}")
            if len(names) > 20:
                print(f"    ... and {len(names) - 20} more")

    print("\n" + "=" * 80)
    print("  EXPLORATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
