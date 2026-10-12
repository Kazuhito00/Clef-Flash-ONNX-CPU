# -*- coding: utf-8 -*-
"""Request validation, option order, span bookkeeping and answer arithmetic (clef_decide.protocol / runtime). No model."""
import pytest

from clef_decide.protocol import PREFIX, SCHEMA, SUFFIX, choice, encode_request, noul, question_options, score, validate
from clef_decide.runtime import _answer


class CharTokenizer:
    """One id per character: enough to check that the spans cut the text where they should."""

    def encode(self, text):
        return [ord(c) for c in text]


def decode(ids):
    return "".join(chr(i) for i in ids)


def test_validate_rejects_bad_requests():
    with pytest.raises(ValueError, match="state"):
        validate({"questions": {"q": noul("x").spec()}})
    with pytest.raises(ValueError, match="at least one"):
        validate({"state": "s", "questions": {}})
    with pytest.raises(ValueError, match="type"):
        validate({"state": "s", "questions": {"q": {"type": "other"}}})
    with pytest.raises(ValueError, match="criteria"):
        validate({"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}})


def test_option_order():
    assert [k for k, _ in question_options(noul("x").spec())] == ["true", "false"]
    assert [k for k, _ in question_options(choice("x", {"b": 1, "a": 2}).spec())] == ["a", "b"]
    assert [k for k, _ in question_options(score("x", ["lo", "mid", "hi"]).spec())] == ["0", "1", "2"]


def test_spans_cover_the_instruction_and_each_option():
    request = {"state": "hello", "questions": {"team": choice("Which team?", {"billing": "charges", "tech": "bugs"}).spec(),
                                               "angry": noul("Is it angry").spec()}}
    enc = encode_request(CharTokenizer(), request)
    text = decode(enc.input_ids)
    assert text.startswith(PREFIX + "hello" + SCHEMA) and text.endswith(SUFFIX)
    assert [text[a:b] for a, b in enc.question_spans] == ["Which team?", "Is it angry"]
    assert len(enc.option_spans) == 4 and enc.option_question == (0, 0, 1, 1)
    assert text[slice(*enc.option_spans[0])] == '{"description":"charges","option_id":"billing"}'
    assert enc.option_ids == (("billing", "tech"), ("true", "false")) and enc.question_types == (1, 0)


def test_image_ids_go_between_the_prefix_and_the_state():
    request = {"state": "s", "questions": {"q": noul("x").spec()}}
    enc = encode_request(CharTokenizer(), request, image_ids=[1, 2, 3])
    assert enc.input_ids[len(PREFIX):len(PREFIX) + 4] == (1, 2, 3, ord("s"))


def test_too_long_request_is_refused_not_truncated():
    with pytest.raises(ValueError, match="at most"):
        encode_request(CharTokenizer(), {"state": "x" * 200, "questions": {"q": noul("x").spec()}}, max_tokens=100)


def test_answer_arithmetic():
    assert _answer({"type": "noul"}, {"true": 0.73214, "false": 0.26786}) == {"type": "noul", "noul": 0.7321}
    got = _answer({"type": "choice", "criteria": {"a": "", "b": ""}}, {"a": 0.2, "b": 0.8})
    assert got["choice"] == "b" and got["confidence"] == 0.8
    got = _answer({"type": "score", "criteria": ["lo", "mid", "hi"]}, {"0": 0.1, "1": 0.2, "2": 0.7})
    assert got["score"] == pytest.approx(1.6) and got["legend"] == {"0": "lo", "1": "mid", "2": "hi"}
