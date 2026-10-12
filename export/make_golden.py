# -*- coding: utf-8 -*-
"""Run Cloudflare's own code on export/cases.py and write artifacts/golden.json.

The oracle is the upstream ``joint_schema_model.py`` (their tokenizer, Qwen3.5 backbone and joint schema head)
loaded in bf16 on CPU -- fp32 would need ~36 GB. This is the only script that needs torch; run it with `uv run --group export`
and with nothing else holding the RAM (the model takes ~18 GB).

    uv run --group export export/make_golden.py

Per case it records the token ids, the spans, the raw option logits and the probabilities. The file is plain JSON.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.cases import CASES  # noqa: E402

MODEL_DIR = ROOT / "models" / "onnx"
OUT = ROOT / "artifacts" / "golden.json"


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
    tokenizer = processor.tokenizer
    golden = {"source": {"model": "Cloudflare/clef-flash", "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
                         "dtype": "bfloat16", "device": "cpu", "torch": torch.__version__},
              "cases": {}}
    for name, request in CASES.items():
        t = time.time()
        with torch.inference_mode():
            enc = upstream.encode_record(tokenizer, request, max_length=16384, processor=processor)
            batch = upstream.collate_records([enc], tokenizer.pad_token_id, torch.device("cpu"))
            logits = model(batch)[0]
        flat = [float(x) for q in logits for x in q.float().tolist()]
        probs = [[float(x) for x in q.float().softmax(-1).tolist()] for q in logits]
        golden["cases"][name] = {
            "request": request,
            "input_ids": list(enc.input_ids),
            "questions": [{"id": q.question_id, "type": q.question_type, "span": list(q.question_span),
                           "options": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
                          for q in enc.questions],
            "logits": flat, "probabilities": probs,
        }
        print("%-14s %4d tokens  %5.1f s  %s" % (name, len(enc.input_ids), time.time() - t,
                                                  ["%.3f" % max(p) for p in probs]), flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")   # keep partial progress
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
