# -*- coding: utf-8 -*-
"""clef-flash (Cloudflare, 9B) typed decisions on onnxruntime + numpy. Text, and images with ``load_vision=True``.

One ONNX graph holds the Qwen3.5-9B backbone AND the joint schema head, so a
decision is exactly one ``session.run``: token ids and spans in, one logit per
option out, then a softmax per question.

The weights are bf16 and live OUTSIDE the graph, as byte ranges of Cloudflare's own
safetensors files (``models/onnx/*.safetensors``). The graph widens them to fp32 with
Cast nodes at run time. If onnxruntime folded those Casts at load time the weights
would be fp32 in memory (~36 GB), more than a 32 GB machine has, so constant folding is
switched off by default (``keep_bf16=True``).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import onnxruntime as ort

from .device import CPU, GPU_PROVIDERS, GpuUnavailableError, check_used, resolve_providers

from .protocol import (NOUL_CRITERIA, PREFIX, QUESTION_TYPES, SCHEMA, SUFFIX, Encoded, Question,
                       encode_request, validate)
from .tokenizer import QwenBPE

__all__ = ["Decider", "GpuUnavailableError", "default_model_dir", "gpu_providers"]

MODEL_DIR_ENV = "CLEF_MODEL_DIR"


def default_model_dir() -> pathlib.Path:
    """``$CLEF_MODEL_DIR`` if set, else ``models/onnx`` of this checkout (where ``download_model.py`` puts the files)."""
    env = os.environ.get(MODEL_DIR_ENV)
    return pathlib.Path(env) if env else pathlib.Path(__file__).resolve().parents[1] / "models" / "onnx"


def gpu_providers() -> List[str]:
    """GPU execution providers available in this onnxruntime build, CPU last (for callers that pass ``providers=``)."""
    available = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS if p in available] + [CPU]


def _uses_cuda(providers: Optional[Sequence[str]]) -> bool:
    return bool(providers) and any("CUDA" in str(p) for p in providers)


def _create_session(path, options: ort.SessionOptions, providers: Optional[Sequence[str]] = None,
                    require_gpu: bool = False, **kwargs) -> ort.InferenceSession:
    """``providers`` None = CPU only. If a GPU provider was asked for but the session cannot use it (out of memory, missing
    DLLs, a build without it) this warns and runs on the CPU, or raises ``GpuUnavailableError`` under ``require_gpu``."""
    wanted = list(providers) if providers else [CPU]
    name = pathlib.Path(str(path)).name
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=wanted, **kwargs)
    except Exception as exc:
        if wanted == [CPU]:
            raise
        detail = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
        if require_gpu:
            raise GpuUnavailableError("%s could not load %s: %s" % (wanted[0], name, detail)) from exc
        print("warning: %s could not load %s (%s); using the CPU" % (wanted[0], name, detail), file=sys.stderr)
        return ort.InferenceSession(str(path), sess_options=options, providers=[CPU], **kwargs)
    check_used(wanted, session.get_providers(), what=name, require_gpu=require_gpu)
    return session


GRAPH = "model-fp32.onnx"
GRAPH_VL = "model-fp32-vl.onnx"   # the same graph with image_embeds / image_positions / mrope_positions inputs
CHUNK = 64          # the graph wants the sequence length padded to a multiple of 64


class Decider:
    """Load once, then call :meth:`decide` / :meth:`predict`.

    >>> d = Decider("models/onnx")
    >>> d.decide("I was charged twice.", {"team": choice("Which team?", {"billing": "charges", "tech": "bugs"})})
    """

    def __init__(
        self,
        model_dir: "str | pathlib.Path | None" = None,
        intra_op_threads: Optional[int] = None,
        keep_bf16: bool = True,
        max_tokens: Optional[int] = None,
        load_vision: bool = False,
        providers: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        require_gpu: bool = False,
    ) -> None:
        """``load_vision=True`` runs EVERY request (text too) through the VL graph and loads the vision tower; the two
        graphs reference the same 18 GB of weights, so only one of them is opened (a second would double the memory).
        ``device``: "cpu" (default) | "cuda" | "auto", see device.py; an explicit onnxruntime list in
        ``providers`` (e.g. ``gpu_providers()``) wins over it. ``require_gpu``: raise instead of falling back to the CPU."""
        if providers is None and (device or require_gpu):
            providers = resolve_providers(device or "cuda", require_gpu=require_gpu)
        root = pathlib.Path(model_dir) if model_dir else default_model_dir()
        graph = root / (GRAPH_VL if load_vision else GRAPH)
        if not graph.exists():
            raise FileNotFoundError("%s not found; run `uv run download_model.py` (or export/prepare_model.py)" % graph)
        self.layout = json.loads((root / "flash" / "decision.json").read_text(encoding="utf-8")) \
            if (root / "flash" / "decision.json").exists() else json.loads((root / "decision.json").read_text(encoding="utf-8"))
        self._check_layout()
        self.pad_id = int(self.layout["pad"])
        self.max_tokens = int(max_tokens or self.layout["max_tokens"])
        self.tokenizer = QwenBPE(str(root / "tokenizer.json"))

        options = ort.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        kwargs: Dict[str, Any] = {}
        if keep_bf16:
            options.enable_cpu_mem_arena = False
            kwargs["disabled_optimizers"] = ["ConstantFolding"]
        if _uses_cuda(providers):
            # onnxruntime 1.31 + CUDA: with graph optimization on, the nodes left on the CPU get a Memcpy node whose copy back fails
            # (`MemcpyToHost ... CUDA failure 1: invalid argument`, every request). Switching the optimizers off fixes it, and the CUDA
            # output then matches the CPU output exactly (A100 40 GB, onnxruntime-gpu 1.31.0; see README "GPUで動かす"). CPU is untouched.
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = _create_session(graph, options, providers, require_gpu, **kwargs)
        self.vision = None
        if load_vision:
            from .image import VisionEncoder
            self.vision = VisionEncoder(root, intra_op_threads, providers=providers, require_gpu=require_gpu)

    @property
    def provider(self) -> str:
        """The execution provider the main graph actually runs on."""
        return self.session.get_providers()[0]

    @property
    def providers_used(self) -> dict:
        """Providers onnxruntime reports per session: ``{"graph": [...]}`` and, with vision, ``"vision"``."""
        used = {"graph": list(self.session.get_providers())}
        if self.vision is not None:
            used["vision"] = list(self.vision.session.get_providers())
        return used

    # ------------------------------------------------------------------ layout

    def _check_layout(self) -> None:
        """The prompt strings are in decision.json (written by the exporter); they must equal the ones this port uses."""
        lay = self.layout
        if lay.get("layout") != "clef-joint-v1":
            raise ValueError("unsupported layout %r" % (lay.get("layout"),))
        p = lay["prompt"]
        mine = {"prefix": PREFIX, "schema": SCHEMA, "suffix": SUFFIX}
        for key, want in mine.items():
            if p[key] != want:
                raise ValueError("decision.json prompt.%s differs from this port: %r" % (key, p[key]))
        if lay["type_index"] != QUESTION_TYPES:
            raise ValueError("question type indices differ")
        if lay["noul_criteria"] != NOUL_CRITERIA:
            raise ValueError("noul criteria differ")

    # ------------------------------------------------------------------ running

    def encode(self, request: Mapping[str, Any], image_ids: Sequence[int] = ()) -> Encoded:
        return encode_request(self.tokenizer, request, self.max_tokens, image_ids)

    def logits(self, enc: Encoded, extra_chunks: int = 0, image_features: Optional[np.ndarray] = None,
               image_grid_thw: Optional[np.ndarray] = None) -> np.ndarray:
        """Raw option logits, shape [options]. ``extra_chunks`` pads further (the result must not change)."""
        n = len(enc.input_ids)
        padded = -(-n // CHUNK) * CHUNK + extra_chunks * CHUNK
        ids = np.full((1, padded), self.pad_id, dtype=np.int64)
        ids[0, :n] = enc.input_ids
        feeds = {
            "input_ids": ids,
            "token_positions": np.arange(n, dtype=np.int64),
            "question_spans": np.asarray(enc.question_spans, dtype=np.int64).reshape(-1, 2),
            "question_types": np.asarray(enc.question_types, dtype=np.int64),
            "option_spans": np.asarray(enc.option_spans, dtype=np.int64).reshape(-1, 2),
            "option_question": np.asarray(enc.option_question, dtype=np.int64),
        }
        if self.vision is not None:                       # the VL graph: three more inputs (N = 0 reproduces the text graph)
            from .image import IMAGE_TOKEN_ID, position_ids
            if image_features is None or not len(image_features):
                slots = np.zeros((0,), dtype=np.int64)
                feats = np.zeros((0, 4096), dtype=np.float32)
                pos = np.broadcast_to(np.arange(n, dtype=np.int64), (3, n))
            else:
                slots = np.flatnonzero(np.asarray(enc.input_ids) == IMAGE_TOKEN_ID).astype(np.int64)
                feats = image_features.astype(np.float32)
                pos = position_ids(enc.input_ids, image_grid_thw)
            mrope = np.zeros((3, padded), dtype=np.int64)
            mrope[:, :n] = pos
            mrope[:, n:] = np.arange(padded - n, dtype=np.int64) + int(pos.max()) + 1     # padding: any increasing values
            feeds.update({"image_embeds": feats, "image_positions": slots, "mrope_positions": mrope})
        elif image_features is not None and len(image_features):
            raise ValueError("this Decider was built without a vision encoder; use Decider(..., load_vision=True)")
        return self.session.run(["logits"], feeds)[0]

    def decide(self, state: Any, questions: Mapping[str, Any], images: Sequence[Any] = (), **kwargs) -> Dict[str, Dict[str, Any]]:
        """Answers in the System One shape: ``{id: {"type": ..., "choice"/"noul"/"score": ..., "probabilities": ...}}``."""
        return self.predict({"state": state, "questions": questions, "images": list(images)}, **kwargs)["answers"]

    def predict(self, request: Mapping[str, Any], extra_chunks: int = 0) -> Dict[str, Any]:
        """A Jev / System One ``/v1/systemone`` request body (state + questions) -> a response body."""
        specs = {qid: (q.spec() if isinstance(q, Question) else dict(q)) for qid, q in request["questions"].items()}
        req = {"state": request["state"], "questions": specs}
        validate(req)
        images = list(request.get("images") or [])
        feats = grid = None
        image_ids: Sequence[int] = ()
        if images:
            if self.vision is None:
                raise ValueError("this Decider was built without a vision encoder; use Decider(..., load_vision=True)")
            from .image import image_prefix_text, prepare_images
            prepared = prepare_images(images, self.vision.config)
            feats, grid = self.vision.encode(prepared), prepared.image_grid_thw
            image_ids = self.tokenizer.encode(image_prefix_text(prepared.token_counts))
        enc = self.encode(req, image_ids)
        flat = self.logits(enc, extra_chunks, feats, grid)
        answers: Dict[str, Dict[str, Any]] = {}
        start = 0
        for qid, ids in zip(enc.question_ids, enc.option_ids):
            chunk = flat[start:start + len(ids)].astype(np.float64)
            start += len(ids)
            probs = np.exp(chunk - chunk.max())
            probs /= probs.sum()
            answers[qid] = _answer(specs[qid], dict(zip(ids, probs.tolist())))
        return {"model": request.get("model", "clef-flash"), "answers": answers,
                "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}


def _answer(question: Mapping[str, Any], p: Mapping[str, float]) -> Dict[str, Any]:
    """The same arithmetic as Cloudflare's ``systemone_answer``."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": round(p["true"], 4)}
    if kind == "choice":
        options = [str(o) for o in question["criteria"]]
        best = max(options, key=p.__getitem__)
        return {"type": "choice", "choice": best, "confidence": round(p[best], 4),
                "probabilities": {o: round(p[o], 4) for o in options}}
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {"type": "score", "score": round(sum(i * p[l] for i, l in enumerate(levels)), 4),
            "confidence": round(max(p[l] for l in levels), 4),
            "legend": dict(zip(levels, question["criteria"])),
            "probabilities": {l: round(p[l], 4) for l in levels}}
