# -*- coding: utf-8 -*-
"""Put a runnable model under models/.

Every repository in this family exposes the same entry point; what it does
differs. For clef-flash the graph is ollaya's published ONNX export, which holds
NO weights: each weight is an external-data reference (file + byte offset) into
Cloudflare's own safetensors files. So this

  1. downloads Cloudflare/clef-flash at the revision the graph was exported from
     (19 GB: four bf16 shards + the joint head), and checks their sha256;
  2. downloads the graph (12 MB) from ollaya-dev/clef;
  3. rewrites the graph's weight references from ollaya's blob names
     ("sha256-<hash>") to the real file names, in one pass over the protobuf.
     No weight is copied or converted.

    python export/prepare_model.py --list
    python export/prepare_model.py
    python export/prepare_model.py --skip-hash        # trust the sizes, skip the 19 GB hash

No git and no git-lfs: plain HTTPS from Hugging Face's resolve endpoint, resumable.
Needs only the standard library plus `onnx` for step 3 (requirements-export.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.download import fetch_files  # noqa: E402

#: The graph's offsets are only valid for exactly this revision of the weights.
WEIGHTS_REPO = "Cloudflare/clef-flash"
WEIGHTS_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
#: ollaya's export of that revision.
GRAPH_REPO = "ollaya-dev/clef"
GRAPH_REVISION = "85b1c5cd90df6308ea95990c3e8a4f207454dad0"

MODEL_DIR = ROOT / "models" / "onnx"

#: The weight files. Their sha256 is NOT written here: it is read from the Hub's file listing of the pinned revision
#: (LFS oid) and checked against the file on disk, and the graph names its weights by the same hashes.
WEIGHT_FILES = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "joint_head.safetensors",
]
#: Small files the runtime / oracle read.
SUPPORT_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json", "joint_head_config.json",
                 "processor_config.json", "generation_config.json", "chat_template.jinja",
                 "model.safetensors.index.json", "joint_schema_model.py", "LICENSE"]
GRAPH_FILES = ["flash/model-fp32.onnx", "flash/decision.json", "flash/calibration.json"]
GRAPH_NAME = "model-fp32.onnx"


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 24)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def verify_weights(skip_hash: bool) -> dict:
    """sha256 -> file name for every weight file, after checking each one."""
    full = _lfs_oids()
    by_hash = {}
    for name in WEIGHT_FILES:
        path = MODEL_DIR / name
        want = full[name]
        if not skip_hash:
            print("  sha256 %-40s " % name, end="", flush=True)
            got = sha256_of(path)
            print("ok" if got == want else "MISMATCH")
            if got != want:
                raise SystemExit("%s does not match revision %s (got %s)" % (name, WEIGHTS_REVISION[:8], got))
        by_hash["sha256-" + want] = name
    return by_hash


def _lfs_oids() -> dict:
    """The sha256 of every weight file, straight from the Hub's file listing for the pinned revision."""
    url = "https://huggingface.co/api/models/%s/tree/%s?recursive=true" % (WEIGHTS_REPO, WEIGHTS_REVISION)
    request = urllib.request.Request(url, headers={"User-Agent": "clef-decide-onnx/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        listing = json.load(response)
    oids = {item["path"]: item["lfs"]["oid"] for item in listing if item.get("lfs")}
    return {name: oids[name] for name in WEIGHT_FILES}


def relink_graph(by_hash: dict) -> int:
    """Point the graph's external-data references at the real file names (offsets and lengths stay)."""
    import onnx

    source = MODEL_DIR / "flash" / GRAPH_NAME
    model = onnx.load(str(source), load_external_data=False)
    changed = 0
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                if entry.value not in by_hash:
                    raise SystemExit("graph refers to an unknown blob: %s" % entry.value)
                entry.value = by_hash[entry.value]
                changed += 1
    out = MODEL_DIR / GRAPH_NAME
    onnx.save(model, str(out))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show what will be downloaded")
    parser.add_argument("--skip-hash", action="store_true", help="do not hash the 19 GB of weights")
    args = parser.parse_args()

    if args.list:
        print("weights   %s @ %s" % (WEIGHTS_REPO, WEIGHTS_REVISION[:8]))
        for name in WEIGHT_FILES:
            print("   ", name)
        print("graph     %s @ %s" % (GRAPH_REPO, GRAPH_REVISION[:8]))
        for name in GRAPH_FILES:
            print("   ", name)
        print("\n~19 GB in total; the graph is 12 MB. Everything lands under models/onnx/.")
        return 0

    print("weights: %s @ %s" % (WEIGHTS_REPO, WEIGHTS_REVISION[:8]))
    fetch_files(WEIGHTS_REPO, list(WEIGHT_FILES) + SUPPORT_FILES, MODEL_DIR, revision=WEIGHTS_REVISION)
    print("graph: %s @ %s" % (GRAPH_REPO, GRAPH_REVISION[:8]))
    fetch_files(GRAPH_REPO, GRAPH_FILES, MODEL_DIR, revision=GRAPH_REVISION)
    print("checking the weights ...")
    by_hash = verify_weights(args.skip_hash)
    n = relink_graph(by_hash)
    print("relinked %d weight references -> models/onnx/%s" % (n, GRAPH_NAME))
    print("\ndone. try:  python demo_inference_text.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
