# -*- coding: utf-8 -*-
"""The pure-Python Qwen byte-level BPE on a tiny tokenizer.json (the real one is checked by verify/check_tokenizer_fuzz.py)."""
from clef_decide.tokenizer import QwenBPE


def test_merges_follow_rank_order(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert tok.encode("the") == [tok.vocab["the"]]
    assert tok.encode("and") == [tok.vocab["and"]]
    assert tok.encode(" the") == [tok.vocab["Ġthe"]]


def test_bytes_outside_ascii_become_byte_tokens(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert len(tok.encode("é")) == 2                    # two UTF-8 bytes, no merge for them
    assert tok.encode("") == []


def test_added_tokens_are_cut_out_of_the_text(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    ids = tok.encode("the<|image_pad|>the")
    assert ids[0] == tok.vocab["the"] and ids[2] == tok.vocab["the"] and len(ids) == 3
