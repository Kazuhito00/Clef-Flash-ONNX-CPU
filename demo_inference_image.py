# -*- coding: utf-8 -*-
"""Typed decisions about pictures with clef-flash (Cloudflare, 9B), on onnxruntime + numpy + Pillow.

    python demo_inference_image.py                      # the bundled sample (artifacts/images/sample_chart.png) and two built-in questions
    python demo_inference_image.py photo.jpg --noul "dog=There is a dog in the picture"
    python demo_inference_image.py a.png b.png --state "Checkout page." --choice "next=retry,back,none"

Up to 4 images (PNG, JPEG, WebP, GIF); each is scaled down so its longer side is at most 448 px. They go right after
STATE: and the backbone reads them with Qwen3.5's own vision tower, then every question is answered in one pass.
Needs vision_encoder.onnx and model-fp32-vl.onnx next to the other model files (download_model.py fetches them;
export/export_vision.py and export/add_image_inputs.py make them). The VL graph is used for the whole run; the first call reads ~18 GB of weights.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from clef_decide import Decider  # noqa: E402
from clef_decide.device import GpuUnavailableError, add_device_args, describe  # noqa: E402
from clef_decide.protocol import choice, noul  # noqa: E402
from demo_inference_text import parse_questions, show  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
SAMPLE = ROOT / "artifacts" / "images" / "sample_chart.png"     # a bar chart drawn by export/cases_image.py (no third-party content)
SAMPLE_QUESTIONS = {
    "tallest": choice("Which bar is the tallest?", {"first": "leftmost bar", "second": "second from the left",
                                                    "third": "third from the left", "fourth": "rightmost bar"}),
    "bar_chart": noul("The picture is a bar chart"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="up to 4 pictures (default: the bundled sample chart)")
    ap.add_argument("--state", default="")
    ap.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    ap.add_argument("--score", action="append", metavar="NAME=low,...,high")
    ap.add_argument("--noul", action="append", metavar="NAME=statement")
    ap.add_argument("--instructions", default="")
    ap.add_argument("--model", default=None, help="model folder (default: $CLEF_MODEL_DIR, else models/onnx)")
    ap.add_argument("--threads", type=int, default=8)
    add_device_args(ap)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    questions = parse_questions(args)
    if not args.images:
        args.images = [str(SAMPLE)]
        questions = questions or SAMPLE_QUESTIONS
    if not questions:
        ap.error("give at least one --choice, --score or --noul")
    try:
        decider = Decider(ROOT / args.model if args.model else None, intra_op_threads=args.threads, load_vision=True,
                          device=args.device, require_gpu=args.require_gpu)
    except GpuUnavailableError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    started = time.perf_counter()
    answers = decider.decide(args.state, questions, images=args.images)
    elapsed = time.perf_counter() - started
    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0
    print("images: %s\n" % ", ".join(args.images))
    for name, answer in answers.items():
        show(name, answer)
    print("\n%d questions, one pass, %.1f s (vision tower included)" % (len(answers), elapsed))
    print("providers used -> %s" % describe(decider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
