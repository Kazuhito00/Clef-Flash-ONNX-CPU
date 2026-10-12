# -*- coding: utf-8 -*-
"""Gate: the image path (Pillow + onnxruntime + numpy) against Cloudflare's own code (artifacts/golden_image.json).

Per case: our token ids (expanded image placeholders between the prefix and the state) must equal the oracle's exactly,
then the option logits and decisions must agree within the bf16 reference's tolerance (see verify.py). Loads the 9B model
and the vision tower: ~15 GB of RAM, nothing else should be running.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_decide.runtime import default_model_dir  # noqa: E402
sys.path.insert(0, str(ROOT / "export"))
from cases_image import images  # noqa: E402
from clef_decide import Decider  # noqa: E402
from clef_decide.image import image_prefix_text, prepare_images  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.25, help="max |logit difference| (bf16 reference, plus the vision tower)")
    ap.add_argument("--tie", type=float, default=0.02)
    args = ap.parse_args()

    golden = json.loads((ROOT / "artifacts" / "golden_image.json").read_text(encoding="utf-8"))
    pics = images()
    d = Decider(default_model_dir(), intra_op_threads=args.threads, load_vision=True)
    bad = 0
    worst = 0.0
    for name, case in golden["cases"].items():
        req = case["request"]
        prepared = prepare_images([pics[n] for n in req["images"]], d.vision.config)
        feats = d.vision.encode(prepared)
        ids = d.tokenizer.encode(image_prefix_text(prepared.token_counts))
        enc = d.encode({"state": req["state"], "questions": req["questions"]}, ids)
        problems = []
        if prepared.image_grid_thw.tolist() != case["image_grid_thw"]:
            problems.append("grid")
        if list(enc.input_ids) != case["input_ids"]:
            problems.append("input ids")
        got = d.logits(enc, 0, feats, prepared.image_grid_thw)
        want = np.asarray(case["logits"], dtype=np.float64)
        diff = float(np.abs(got - want).max()) if got.shape == want.shape else float("inf")
        worst = max(worst, diff)
        start = 0
        for q, ref in zip(case["questions"], case["probabilities"]):
            n = len(q["option_ids"])
            mine = got[start:start + n].astype(np.float64)
            start += n
            p = np.exp(mine - mine.max())
            p /= p.sum()
            top2 = sorted(ref, reverse=True)[:2]
            if int(np.argmax(p)) != int(np.argmax(ref)) and not (len(top2) > 1 and top2[0] - top2[1] < args.tie):
                problems.append("%s: decision differs" % q["id"])
        if diff > args.tol:
            problems.append("logits")
        bad += bool(problems)
        print("%-18s %4d tok  max|dlogit| %.4f  %s" % (name, len(enc.input_ids), diff, "ok" if not problems else "DIFFERS: " + ", ".join(problems)), flush=True)
    print("\nworst |dlogit| %.4f; failures %d of %d" % (worst, bad, len(golden["cases"])))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
