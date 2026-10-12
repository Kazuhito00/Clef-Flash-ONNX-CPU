# -*- coding: utf-8 -*-
"""Gate: our token ids and spans equal the reference's, case by case, exactly.

Needs only the runtime (no model is loaded). The reference ids come from artifacts/golden.json, which
export/make_golden.py wrote by running Cloudflare's own ``encode_record``.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_decide.runtime import default_model_dir  # noqa: E402
from clef_decide.protocol import encode_request  # noqa: E402
from clef_decide.tokenizer import QwenBPE  # noqa: E402


def main() -> int:
    golden = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))
    tok = QwenBPE(str(default_model_dir() / "tokenizer.json"))
    bad = 0
    for name, case in golden["cases"].items():
        enc = encode_request(tok, case["request"], max_tokens=16384)
        want = case["questions"]
        problems = []
        if list(enc.input_ids) != case["input_ids"]:
            problems.append("input_ids")
        if [list(s) for s in enc.question_spans] != [q["span"] for q in want]:
            problems.append("question spans")
        if [list(s) for s in enc.option_spans] != [s for q in want for s in q["options"]]:
            problems.append("option spans")
        if [list(i) for i in enc.option_ids] != [q["option_ids"] for q in want]:
            problems.append("option ids/order")
        if list(enc.question_types) != [q["type"] for q in want]:
            problems.append("types")
        print("%-14s %4d tokens  %s" % (name, len(enc.input_ids), "ok" if not problems else "DIFFERS: " + ", ".join(problems)))
        bad += bool(problems)
    print("\nfailures: %d of %d" % (bad, len(golden["cases"])))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
