# -*- coding: utf-8 -*-
"""Gate: the logits do not depend on how far the sequence is padded.

The graph wants the length padded to a multiple of 64 and uses the real token count (token_positions) for
everything that matters. Running the same request padded by one more 64-token chunk must give the same logits. This
catches a loop length baked in at export, and padding that leaks into the head.

Loads the 9B model (~14 GB of RAM).
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_decide.runtime import default_model_dir  # noqa: E402
from clef_decide import Decider, choice, noul  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    d = Decider(default_model_dir(), intra_op_threads=args.threads)
    request = {"state": "I was charged twice for my Pro plan.",
               "questions": {"team": choice("Which team?", {"billing": "charges", "technical": "bugs"}).spec(),
                             "angry": noul("The customer is angry").spec()}}
    enc = d.encode(request)
    base = d.logits(enc)
    worst = 0.0
    for extra in (1, 2):
        other = d.logits(enc, extra_chunks=extra)
        diff = float(np.abs(base - other).max())
        worst = max(worst, diff)
        print("padded by %d extra chunk(s): max |dlogit| = %.3e" % (extra, diff), flush=True)
    print("\nworst %.3e" % worst)
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
