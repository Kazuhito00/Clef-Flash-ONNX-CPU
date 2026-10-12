# -*- coding: utf-8 -*-
"""Which onnxruntime execution providers to ask for, and what was really used.

No onnxruntime session is created here, so the logic is testable without a model.

    --device cpu     CPUExecutionProvider only (the default)
    --device cuda    CUDAExecutionProvider, then CPUExecutionProvider
    --device auto    CUDA when this onnxruntime build has it, else CPU
    --require-gpu    an error (not a silent CPU fallback) when the graph does not end up on a GPU provider

onnxruntime can drop a provider it cannot load and carry on with the CPU. A GPU request therefore always checks the
providers the session reports (``InferenceSession.get_providers()``) and warns, or raises under ``--require-gpu``.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DEVICES = ("cpu", "cuda", "auto")
#: Providers that count as "a GPU" for --require-gpu.
GPU_PROVIDERS = (CUDA, "DmlExecutionProvider")


class GpuUnavailableError(RuntimeError):
    """A GPU was required (``require_gpu``) but is not available, or the graph did not run on it."""


def available_providers() -> List[str]:
    import onnxruntime as ort
    return list(ort.get_available_providers())


def resolve_providers(device: str = "cpu", available: Optional[Sequence[str]] = None,
                      require_gpu: bool = False) -> List[str]:
    """Providers to pass to ``InferenceSession`` for ``device``.

    Raises ``GpuUnavailableError`` for ``require_gpu`` when CUDA is missing. ``device="cuda"`` without CUDA warns and
    returns the CPU list (so a plain request never fails, but it is never silent either).
    """
    if device not in DEVICES:
        raise ValueError("device must be one of %s, not %r" % (", ".join(DEVICES), device))
    if available is None:
        available = available_providers()
    if device == "cpu":
        if require_gpu:
            raise GpuUnavailableError("require_gpu contradicts device='cpu'")
        return [CPU]
    if CUDA in available:
        return [CUDA, CPU]
    if require_gpu or device == "cuda":
        msg = ("CUDAExecutionProvider is not available in this onnxruntime build (providers: %s). "
               "Install onnxruntime-gpu instead of onnxruntime (the two cannot be installed together)."
               % ", ".join(available))
        if require_gpu:
            raise GpuUnavailableError(msg)
        print("warning: %s Running on the CPU." % msg, file=sys.stderr)
    return [CPU]


def check_used(requested: Sequence[str], used: Sequence[str], what: str = "graph", require_gpu: bool = False) -> bool:
    """True when the session runs on the first requested provider.

    Warns (or raises under ``require_gpu``) when a GPU provider was requested but the session reports something else.
    """
    wants_gpu = bool(requested) and requested[0] in GPU_PROVIDERS
    ok = bool(used) and used[0] == requested[0]
    if wants_gpu and not ok:
        msg = "%s was asked to run on %s but onnxruntime reports %s" % (what, requested[0], list(used))
        if require_gpu:
            raise GpuUnavailableError(msg)
        print("warning: %s; it runs on the CPU" % msg, file=sys.stderr)
    return ok


def add_device_args(parser) -> None:
    """``--device`` and ``--require-gpu`` for a demo's argparse parser."""
    parser.add_argument("--device", choices=DEVICES, default="cpu",
                        help="cpu (default) | cuda (CUDA, then CPU) | auto (CUDA if this onnxruntime has it)")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail instead of silently running on the CPU when the model does not end up on a GPU")


def describe(decider) -> str:
    """The providers really in use, e.g. ``graph: CUDAExecutionProvider, CPUExecutionProvider; vision: ...``."""
    return "; ".join("%s: %s" % (k, ", ".join(v)) for k, v in decider.providers_used.items())
