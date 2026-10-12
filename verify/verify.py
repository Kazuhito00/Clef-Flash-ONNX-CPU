# -*- coding: utf-8 -*-
"""Gate: option logits and typed answers from the ONNX graph vs Cloudflare's own code (artifacts/golden.json).

The reference runs the backbone in bf16; the graph keeps bf16 weights but computes in fp32 (as ollaya's export does),
so the logits are close, not equal. What must hold:
  * every question gets the same decision (argmax over its options), and
  * the logits stay within a tolerance. Near-ties (the reference's top-2 probabilities closer than --tie) are
    reported on their own, not hidden.

Loads the 9B model: ~14 GB of RAM, ~15-80 s per request on 8 threads. Run it with nothing else holding the memory.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_decide.runtime import default_model_dir  # noqa: E402
from clef_decide import Decider  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.15, help="max |logit difference| allowed (bf16 reference)")
    ap.add_argument("--tie", type=float, default=0.02, help="top-2 probability gap that counts as a near-tie")
    ap.add_argument("--only", nargs="*", help="case names")
    ap.add_argument("--vl", action="store_true", help="run through the VL graph with no image (must equal the text graph)")
    args = ap.parse_args()

    golden = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))
    decider = Decider(default_model_dir(), intra_op_threads=args.threads, load_vision=args.vl)
    wrong = ties = 0
    worst = 0.0
    worst_p = 0.0
    for name, case in golden["cases"].items():
        if args.only and name not in args.only:
            continue
        enc = decider.encode(case["request"])
        got = decider.logits(enc)
        want = np.asarray(case["logits"], dtype=np.float64)
        diff = float(np.abs(got - want).max())
        worst = max(worst, diff)
        start = 0
        notes = []
        for q, ref_probs in zip(case["questions"], case["probabilities"]):
            n = len(q["option_ids"])
            mine = got[start:start + n].astype(np.float64)
            start += n
            p = np.exp(mine - mine.max())
            p /= p.sum()
            worst_p = max(worst_p, float(np.abs(p - np.asarray(ref_probs)).max()))
            same = int(np.argmax(p)) == int(np.argmax(ref_probs))
            top2 = sorted(ref_probs, reverse=True)[:2]
            tie = len(top2) > 1 and top2[0] - top2[1] < args.tie
            if not same:
                if tie:
                    ties += 1
                    notes.append("%s: near-tie flipped (%.3f vs %.3f)" % (q["id"], top2[0], top2[1]))
                else:
                    wrong += 1
                    notes.append("%s: DECISION DIFFERS" % q["id"])
        flag = "ok " if diff <= args.tol and not any("DIFFERS" in n for n in notes) else "XX "
        print("%s%-14s max|dlogit| %.4f  %s" % (flag, name, diff, "; ".join(notes)), flush=True)
        wrong += flag == "XX " and not any("DIFFERS" in n for n in notes)
    print("\ndecisions differing: %d   near-tie flips: %d   worst |dlogit| %.4f   worst |dprob| %.5f"
          % (wrong, ties, worst, worst_p))
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
