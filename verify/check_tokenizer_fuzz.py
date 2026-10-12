# -*- coding: utf-8 -*-
"""Oracle: the pure-Python Qwen BPE vs the tokenizer Cloudflare's own code uses
(transformers AutoTokenizer, i.e. the Rust `tokenizers` library behind it).

Run with `uv run --group export` (transformers installed). The corpus covers the shapes the clef prompts take,
plus the Unicode classes where Python's `re` and the Rust regex crate are known to disagree.
"""
import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_decide.runtime import default_model_dir  # noqa: E402
from clef_decide.tokenizer import QwenBPE  # noqa: E402

TOKJSON = default_model_dir() / "tokenizer.json"

SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<|object_ref_start|>",
    "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
]

WHITESPACE = [
    " ", " ", " ", "　", "\x0b", "\x0c", "\x1c", "\x1d",
    "\x1e", "\x1f", "​", " ", " ", "﻿", "\x85", " ",
]

BASE = [
    "", " ", "  ", "   ", "\t", "\n", "\r\n", "\n\n\n", " \n ", "a\n\nb",
    "a", "A", "Z", "AA", "ZZ", "The", " the", "the ",
    "don't", "it's", "they're", "we've", "I'm", "we'll", "he'd",
    "DON'T", "It'S", "THEY'RE",
    "CamelCase", "ALLCAPS", "snake_case_name", "kebab-case-name", "mixed123digits",
    "0", "42", "3.14", "1,000,000", "2026-09-28", "v2.0.0", "0x1F",
    "user@example.com", "https://example.com/a/b?c=d&e=f", "www.example.co.jp",
    "@handle", "#hashtag", "50%", "$1,234.56", "¥5,400", "€99", "100°C",
    "naïve café résumé", "Straße Zürich", "İstanbul", "ﬁrefly", "ǅungla",
    "日本語のテキストです", "これはテストです。", "漢字かなカナ", "全角　スペース",
    "한국어 텍스트", "Русский текст", "العربية", "עברית", "ไทย", "हिन्दी",
    "emoji \U0001f389 test", "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 family",
    "\U0001f1ef\U0001f1f5 flag", "a\U0001f389b",
    "égalité", "á combining", "①②③", "ⅷ roman", "㍿ square",
    '{"state":{"cart":3},"total":42}', "[1,2,3]", "```python\nx = 1\n```",
    "a" * 40, "ab" * 30, "x" * 100, " " * 20, "\n" * 10,
    "The export button crashes in Safari but works in Chrome. Not blocking.",
]
BASE += SPECIALS
BASE += WHITESPACE
BASE += ["a%sb" % w for w in WHITESPACE]
BASE += ["a%s%sb" % (w, w) for w in WHITESPACE[:8]]
BASE += ["x %s y" % s for s in SPECIALS]

# the pieces clef tokenizes one by one (see clef_decide/protocol.py)
BASE += [
    "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one "
    "of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n",
    "\n\nSCHEMA FIELDS:\n",
    "\nFIELD 1\nID: team\nTYPE: choice\nINSTRUCTION: ",
    "\nALLOWED OPTIONS:\n", "OPTION 1: ", "OPTION 12: ", "END FIELD\n", "\n",
    '{"description":"charges, refunds","option_id":"billing"}',
    '{"description":"The proposition is true or the answer is yes.","option_id":"true"}',
    '{"option_id":"0"}',
    '{"customer":"Aiko","notes":null,"order":{"id":1042,"items":["lamp","desk"],"status":"shipped"}}',
    '{"description":"請求・返金","option_id":"請求"}',
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
]

ALPHABET = list(
    "abcdeABCDE 0123(),.:;-_|?!@#$%/" + chr(92) + chr(39) + chr(34) + "`~^*+=<>[]{}"
    "éüñçİı"
    "日本語漢字ひらがなカタ"
    "한글Русالعไท"
    "\U0001f389\U0001f44d\U0001f1ef\U0001f1f5‍️"
    " 　  ​﻿\x0b\x0c\x1c\x85"
    "\t\n\r "
    "①ⅷ㍿ﬁǅ́"
)


def random_cases(n, rng):
    out = []
    for _ in range(n):
        k = rng.randint(1, 60)
        out.append("".join(rng.choice(ALPHABET) for _ in range(k)))
    for _ in range(n // 4):
        w = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12)))
        out.append("<|im_start|>user\n%s %s %s<|im_end|>\n" % (w, rng.choice(SPECIALS), w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=4000)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rng = random.Random(20261006)
    mine = QwenBPE(str(TOKJSON))
    ref = AutoTokenizer.from_pretrained(str(default_model_dir()))
    cases = list(BASE) + random_cases(args.n, rng)

    bad = []
    for text in cases:
        got = list(mine.encode(text))
        exp = ref(text, add_special_tokens=False).input_ids
        if got != exp:
            bad.append((text, got, exp))

    print("cases: %d   mismatches: %d" % (len(cases), len(bad)))
    for text, got, exp in bad[:12]:
        print("\n  input : %r" % text)
        print("  got   : %s" % got)
        print("  expect: %s" % exp)
        print("  gotstr: %s" % ref.convert_ids_to_tokens(got))
        print("  expstr: %s" % ref.convert_ids_to_tokens(exp))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
