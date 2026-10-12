# -*- coding: utf-8 -*-
"""Run Cloudflare's own code on picture requests (export/cases_image.py) and write artifacts/golden_image.json.

    uv run --group export export/make_golden_image.py

Like make_golden.py: the oracle is the upstream ``joint_schema_model.py`` (its tokenizer, processor, Qwen3.5 backbone
with the vision tower, joint schema head) loaded in bf16 on CPU, ~18 GB of RAM, so run it with nothing else holding
memory. The pictures are first capped to 448 px on the longer side by this repository's own ``fit_long_side`` (the
deliberate deviation documented in clef_decide/image.py) and then handed to upstream's processor unchanged, so the
oracle and the port see identical pixels. Per question set it records the token ids (with the expanded image
placeholders), the image grids, the raw option logits and the probabilities.
"""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_decide.image import decode_image, fit_long_side  # noqa: E402
from export.cases_image import CASES, images  # noqa: E402

MODEL_DIR = ROOT / "models" / "onnx"
OUT = ROOT / "artifacts" / "golden_image.json"


def load_upstream():
    spec = importlib.util.spec_from_file_location("joint_schema_model", MODEL_DIR / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import torch

    upstream = load_upstream()
    started = time.time()
    model, processor = upstream.load_release_model(MODEL_DIR, device="cpu", dtype=torch.bfloat16)
    print("loaded in %.0f s" % (time.time() - started), flush=True)
    from clef_decide.image import pil_image_processor
    processor.image_processor = pil_image_processor(MODEL_DIR)       # PIL backend, as the Strands vision engine pins it
    tokenizer = processor.tokenizer
    pics = images()
    capped = {k: fit_long_side(decode_image(v), 448) for k, v in pics.items()}
    golden = {"source": {"model": "Cloudflare/clef-flash", "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
                         "dtype": "bfloat16", "device": "cpu", "torch": torch.__version__, "image_long_side": 448},
              "cases": {}}
    for name, case in CASES.items():
        t = time.time()
        request = {"state": case["state"], "questions": case["questions"], "images": [capped[n] for n in case["images"]]}
        with torch.inference_mode():
            enc = upstream.encode_record(tokenizer, request, max_length=16384, processor=processor)
            batch = upstream.collate_records([enc], tokenizer.pad_token_id, torch.device("cpu"))
            logits = model(batch)[0]
        flat = [float(x) for q in logits for x in q.float().tolist()]
        probs = [[float(x) for x in q.float().softmax(-1).tolist()] for q in logits]
        golden["cases"][name] = {
            "request": {"state": case["state"], "questions": case["questions"], "images": case["images"]},
            "input_ids": list(enc.input_ids), "image_grid_thw": enc.media["image_grid_thw"].tolist(),
            "questions": [{"id": q.question_id, "type": q.question_type, "span": list(q.question_span),
                           "options": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
                          for q in enc.questions],
            "logits": flat, "probabilities": probs,
        }
        print("%-18s %4d tokens  %5.1f s  %s" % (name, len(enc.input_ids), time.time() - t,
                                                  ["%.3f" % max(p) for p in probs]), flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
