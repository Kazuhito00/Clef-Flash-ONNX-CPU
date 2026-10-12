# -*- coding: utf-8 -*-
"""First load: does the 9B graph load on this CPU, how much memory does it take, how long is one forward?

    python verify/smoke.py [--threads 8] [--fold]       (--fold lets onnxruntime constant-fold: expect ~2x memory)

Prints the peak working set after load and after the first run. Stops with a non-zero exit when
the load fails; it does not try to be clever about a machine that is out of memory.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def peak_mb() -> float:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(Counters), wt.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fold", action="store_true", help="allow constant folding (default: off, keeps bf16 weights)")
    ap.add_argument("--model", default=None, help="model folder (default: $CLEF_MODEL_DIR, else models/onnx)")
    args = ap.parse_args()

    from clef_decide import Decider, choice, default_model_dir, noul

    t = time.perf_counter()
    d = Decider((ROOT / args.model if args.model else default_model_dir()), intra_op_threads=args.threads, keep_bf16=not args.fold)
    print("load      %6.1f s   peak %7.0f MB" % (time.perf_counter() - t, peak_mb()), flush=True)
    request = {"state": "I was charged twice for my Pro plan.",
               "questions": {"team": choice("Which team should handle this?",
                                            {"billing": "charges, refunds", "technical": "bugs, outages"}),
                             "angry": noul("The customer is angry")}}
    for i in range(2):
        t = time.perf_counter()
        out = d.predict(request)
        print("forward %d %6.1f s   peak %7.0f MB   %d tokens" % (i + 1, time.perf_counter() - t, peak_mb(),
                                                                  out["usage"]["input_tokens"]), flush=True)
    print(out["answers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
