# -*- coding: utf-8 -*-
"""Give ollaya's text graph two extra inputs, so that images can go through it. Pure graph surgery (onnx only):
no weight is touched, nothing is exported from torch.

    python export/add_image_inputs.py            # models/onnx/model-fp32.onnx -> model-fp32-vl.onnx

The text graph embeds ``input_ids`` itself and builds 1-D rotary positions from the sequence length. Qwen3.5's
vision-language use differs in exactly two places, and each is one place in the graph:

1. The token embeddings of the <|image_pad|> positions are replaced by the vision encoder's features.
       embedding' = ScatterND(embedding, [(0, p) for p in image_positions], image_embeds)

2. The rotary angles come from THREE position streams (time, height, width), interleaved over the 32 frequency
   pairs (mrope_section [11, 11, 10], "mrope_interleaved"): frequency k uses the height stream when
   k % 3 == 1 and k <= 31, the width stream when k % 3 == 2 and k <= 29, the time stream otherwise.
       angles[s, k] = mrope_positions[axis(k), s] * inv_freq[k]
   (the graph computes cos/sin ONCE from ``Mul(arange(S)[:, None], inv_freq[None, :])`` and every full-attention layer
   shares it, so replacing that one Mul is enough; the linear-attention layers use no rotary embedding.)

New inputs
    image_embeds     float32 [N, 4096]   vision features, in the order of the image tokens
    image_positions  int64   [N]         sequence index of each image token
    mrope_positions  int64   [3, S]      (time, height, width) position of every token, padded like input_ids

With N = 0 and mrope_positions = arange repeated three times the graph computes exactly what the text graph does
(verify/check_vl_equivalence.py checks that).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "onnx"

HIDDEN = 4096
MROPE_SECTION = (11, 11, 10)
ROTARY_PAIRS = 32


def axis_of_frequency() -> np.ndarray:
    """0 = time, 1 = height, 2 = width, for each of the 32 frequency pairs (Qwen3-VL interleaved layout)."""
    axis = np.zeros(ROTARY_PAIRS, dtype=np.int64)
    for stream, offset in ((1, 1), (2, 2)):
        for k in range(offset, MROPE_SECTION[stream] * 3, 3):
            if k < ROTARY_PAIRS:
                axis[k] = stream
    return axis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="model-fp32.onnx")
    ap.add_argument("--dst", default="model-fp32-vl.onnx")
    args = ap.parse_args()

    model = onnx.load(str(MODEL_DIR / args.src), load_external_data=False)
    graph = model.graph
    nodes = {n.name: n for n in graph.node}

    # ---- 1. image features into the embeddings -------------------------------------------------------------
    widen = nodes["node_embedding:widen"]
    assert widen.op_type == "Cast" and list(widen.output) == ["embedding"], "unexpected embedding node"
    widen.output[0] = "embedding_text"

    graph.input.extend([
        helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["images", HIDDEN]),
        helper.make_tensor_value_info("image_positions", TensorProto.INT64, ["images"]),
        helper.make_tensor_value_info("mrope_positions", TensorProto.INT64, [3, "seq"]),
    ])
    graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "vl_axis1"))
    embed_nodes = [
        helper.make_node("Unsqueeze", ["image_positions", "vl_axis1"], ["vl_pos_col"], name="vl_pos_col"),
        helper.make_node("Shape", ["image_positions"], ["vl_n"], name="vl_n"),
        helper.make_node("ConstantOfShape", ["vl_n"], ["vl_zero_flat"], name="vl_zero_flat",
                         value=numpy_helper.from_array(np.zeros(1, dtype=np.int64))),
        helper.make_node("Unsqueeze", ["vl_zero_flat", "vl_axis1"], ["vl_zero_col"], name="vl_zero_col"),
        helper.make_node("Concat", ["vl_zero_col", "vl_pos_col"], ["vl_indices"], name="vl_indices", axis=1),
        helper.make_node("ScatterND", ["embedding_text", "vl_indices", "image_embeds"], ["embedding"],
                         name="vl_scatter_image_embeds"),
    ]

    # ---- 2. three position streams for the rotary angles -------------------------------------------------
    mul = nodes["node_mul_8"]
    assert mul.op_type == "Mul" and list(mul.input) == ["unsqueeze", "unsqueeze_1"], "unexpected rope node"
    graph.initializer.append(numpy_helper.from_array(axis_of_frequency(), "vl_axis_of_frequency"))
    rope_nodes = [
        helper.make_node("Cast", ["mrope_positions"], ["vl_pos_f"], name="vl_pos_f", to=TensorProto.FLOAT),
        helper.make_node("Gather", ["vl_pos_f", "vl_axis_of_frequency"], ["vl_pos_kS"], name="vl_pos_kS", axis=0),
        helper.make_node("Transpose", ["vl_pos_kS"], ["vl_pos_Sk"], name="vl_pos_Sk", perm=[1, 0]),
    ]
    mul.input[0] = "vl_pos_Sk"

    # keep the graph topologically sorted: the embedding nodes go right after the Cast that widens the embedding,
    # the position nodes right before the Mul that uses them
    order = list(graph.node)
    i_widen, i_mul = order.index(widen), order.index(mul)
    rebuilt = []
    for i, node in enumerate(order):
        if i == i_mul:
            rebuilt.extend(rope_nodes)
        rebuilt.append(node)
        if i == i_widen:
            rebuilt.extend(embed_nodes)
    new_nodes = embed_nodes + rope_nodes
    del graph.node[:]
    graph.node.extend(rebuilt)

    out = MODEL_DIR / args.dst
    onnx.save(model, str(out))
    print("wrote %s  (%d nodes, +%d)" % (out, len(graph.node), len(new_nodes)))
    print("axis of each frequency pair:", axis_of_frequency().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
