# -*- coding: utf-8 -*-
"""The request layout of clef-flash: state + typed questions -> one token sequence with spans.

A port of ``encode_record`` / ``question_options`` / ``render`` from Cloudflare's
``joint_schema_model.py`` (text only). The layout is

    prefix | STATE | schema (one FIELD block per question) | suffix

and every piece is tokenized ON ITS OWN and the id lists are concatenated, so the
ids differ from tokenizing the joined string. The graph needs, besides the ids:

    question_spans   [Q, 2]  instruction tokens of each question, [start, end)
    option_spans     [O, 2]  option tokens of every question, in order
    option_question  [O]     the question each option belongs to
    question_types   [Q]     0 noul, 1 choice, 2 score

and returns one logit per option. ``decision.json`` (from the export) carries the
prompt strings; they are cross-checked against the constants below at load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

__all__ = ["Encoded", "Question", "choice", "encode_request", "noul", "question_options", "render", "score"]

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\nSTATE:\n" % SYSTEM_PROMPT
SCHEMA = "\n\nSCHEMA FIELDS:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


@dataclass(frozen=True)
class Question:
    """One typed question, in the System One request shape."""

    type: str
    instructions: str = ""
    criteria: Any = None

    def spec(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


def choice(instructions: str, criteria: Mapping[str, Any]) -> Question:
    """Pick one named option. ``criteria``: option id -> description."""
    return Question("choice", instructions, dict(criteria))


def score(instructions: str, levels: Sequence[Any]) -> Question:
    """Rate on an ordered scale; the first level is the lowest."""
    return Question("score", instructions, list(levels))


def noul(instructions: str, criteria: Mapping[str, Any] = None) -> Question:
    """A yes/no gate. ``criteria`` may override the wording of "true" / "false"."""
    return Question("noul", instructions, dict(criteria) if criteria else None)


def render(value: Any) -> str:
    """Strings pass through; anything else is compact, key-sorted JSON (no ASCII escaping)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """(option id, description) in the order the model sees them.

    noul: true, false.  choice: ids sorted by code point.  score: "0", "1", ... in level order.
    """
    kind = str(question["type"])
    if kind == "noul":
        criteria = dict(NOUL_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if kind == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class Encoded:
    input_ids: Tuple[int, ...]
    question_ids: Tuple[str, ...]
    question_types: Tuple[int, ...]
    question_spans: Tuple[Tuple[int, int], ...]
    option_spans: Tuple[Tuple[int, int], ...]
    option_question: Tuple[int, ...]
    option_ids: Tuple[Tuple[str, ...], ...]   # per question, in logit order


def validate(request: Mapping[str, Any]) -> Mapping[str, Any]:
    questions = request.get("questions")
    if "state" not in request:
        raise ValueError("state is required")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("at least one question is required")
    for qid, q in questions.items():
        if q.get("type") not in QUESTION_TYPES:
            raise ValueError("%s: type must be noul, choice, or score" % qid)
        if q["type"] != "noul" and not q.get("criteria"):
            raise ValueError("%s: criteria must not be empty" % qid)
    return questions


def encode_request(tokenizer, request: Mapping[str, Any], max_tokens: int = 4096, image_ids: Sequence[int] = ()) -> Encoded:
    """Token ids and spans for a request (``state`` and ``questions``). Raises if it exceeds ``max_tokens``.

    Upstream would truncate the state to fit its own limit; this runtime refuses instead of dropping
    text silently. ``image_ids`` (placeholder tokens of the expanded images plus the newline that follows them, see
    image.image_prefix_text) go between the prefix and the state, where Cloudflare's encoder puts them.
    """
    questions = validate(request)
    enc = tokenizer.encode
    schema_ids: List[int] = list(enc(SCHEMA))
    qids, qtypes, qspans, ospans, oq, oids = [], [], [], [], [], []
    for index, (qid, question) in enumerate(questions.items()):
        schema_ids.extend(enc("\nFIELD %d\nID: %s\nTYPE: %s\nINSTRUCTION: " % (index + 1, qid, question["type"])))
        start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(qid)
        schema_ids.extend(enc(render(instructions)))
        qspans.append((start, len(schema_ids)))
        schema_ids.extend(enc("\nALLOWED OPTIONS:\n"))
        ids = []
        for n, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(enc("OPTION %d: " % (n + 1)))
            o_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(enc(render(semantics)))
            ospans.append((o_start, len(schema_ids)))
            oq.append(index)
            ids.append(option_id)
            schema_ids.extend(enc("\n"))
        schema_ids.extend(enc("END FIELD\n"))
        qids.append(str(qid))
        qtypes.append(QUESTION_TYPES[str(question["type"])])
        oids.append(tuple(ids))

    prefix_ids = list(enc(PREFIX)) + list(image_ids)
    suffix_ids = enc(SUFFIX)
    state_ids = enc(render(request["state"]))
    total = len(prefix_ids) + len(state_ids) + len(schema_ids) + len(suffix_ids)
    if total > max_tokens:
        raise ValueError("request is %d tokens; this graph accepts at most %d (shorten the state or the schema)"
                         % (total, max_tokens))
    shift = len(prefix_ids) + len(state_ids)
    return Encoded(
        input_ids=tuple(prefix_ids + state_ids + schema_ids + suffix_ids),
        question_ids=tuple(qids),
        question_types=tuple(qtypes),
        question_spans=tuple((a + shift, b + shift) for a, b in qspans),
        option_spans=tuple((a + shift, b + shift) for a, b in ospans),
        option_question=tuple(oq),
        option_ids=tuple(oids),
    )
