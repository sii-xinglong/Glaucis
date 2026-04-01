import os

import jax
import jax.numpy as jnp

_IS_TPU_RUNTIME_CACHED: bool | None = None

def exp(x):
    return jnp.exp(x.astype(jnp.float32))

def is_tpu_runtime() -> bool:
    """Return True if the current JAX runtime is on TPU devices.

    Prefer checking actual devices; fall back to default backend if necessary.
    """
    global _IS_TPU_RUNTIME_CACHED
    if _IS_TPU_RUNTIME_CACHED is not None:
        return _IS_TPU_RUNTIME_CACHED
    try:
        devs = jax.devices()
        _IS_TPU_RUNTIME_CACHED = len(devs) > 0 and all(
            d.platform == "tpu" for d in devs
        )
    except Exception:
        _IS_TPU_RUNTIME_CACHED = jax.default_backend() == "tpu"
    return _IS_TPU_RUNTIME_CACHED


def get_interpret() -> bool:
    """Determine the ``interpret`` flag for ``pallas_call``.

    Reads the environment variable ``PALLAS_INTERPRET``.  When set to
    ``"1"`` or ``"true"`` (case-insensitive) interpret mode is enabled;
    every other value (including unset) disables it.
    """
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")
