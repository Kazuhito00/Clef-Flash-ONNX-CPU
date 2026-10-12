# -*- coding: utf-8 -*-
"""Stage a GitHub Release from models/onnx: split what is too big, hash everything, write manifest.json.

    python release/make_release.py --dry-run                 # names, sizes and parts only (reads nothing, writes nothing)
    python release/make_release.py                           # models/onnx -> dist/release/ (reads and hashes ~20 GB)
    python release/make_release.py --version v0.0.0 --max-part-gib 1.9
    python release/make_release.py --models path/to/models   # stage from another checkout's models/

GitHub allows at most 2 GiB per release asset (and 1000 assets per release), so a file larger than ``--max-part-gib`` is cut into
byte parts (``<name>.part000``, ...) that download_model.py joins again and checks against the whole file's sha256. Asset names are
flat, so ``flash/decision.json`` is published as ``flash__decision.json``; manifest.json keeps the real path.
Nothing is uploaded here. Upload dist/release/* yourself, for example:

    gh release create v0.0.0 dist/release/* --title "v0.0.0" --notes-file release/notes.md

What goes in (everything the runtime, the demos and the oracle gates read):
  * the files made here: the two graphs (model-fp32.onnx, model-fp32-vl.onnx; no weights inside), the vision tower
    (vision_encoder.onnx + .onnx.data, 1.8 GB), flash/decision.json, flash/calibration.json;
  * Cloudflare's own files, byte for byte, at the revision the graphs were exported for (17f0b0ad): the four bf16 safetensors
    shards (3.9 - 5.0 GB, each split into parts) + joint_head.safetensors, tokenizer, configs, LICENSE.
The graphs read the weights by byte offset, so the Release must carry exactly those bytes. The same bytes can be fetched from
Hugging Face instead (``python export/prepare_model.py``), which also checks their sha256 against the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK = 1 << 20

#: Paths below models/onnx that are released (the weights come first in the listing).
WEIGHTS = ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors",
           "model-00004-of-00004.safetensors", "joint_head.safetensors"]
FILES = ["model-fp32.onnx", "model-fp32-vl.onnx", "vision_encoder.onnx", "vision_encoder.onnx.data",
         "flash/decision.json", "flash/calibration.json",
         "tokenizer.json", "tokenizer_config.json", "config.json", "joint_head_config.json", "processor_config.json",
         "generation_config.json", "chat_template.jinja", "model.safetensors.index.json", "joint_schema_model.py", "LICENSE"]


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # no second copy of a multi-GB file on the same drive
    except OSError:
        shutil.copy2(src, dst)


def part_sizes(size: int, part_bytes: int) -> list:
    return [min(part_bytes, size - start) for start in range(0, size, part_bytes)]


def split(src: pathlib.Path, out_dir: pathlib.Path, stem: str, part_bytes: int) -> list:
    parts = []
    with open(src, "rb") as handle:
        index = 0
        while True:
            name = "%s.part%03d" % (stem, index)
            digest, written = hashlib.sha256(), 0
            with open(out_dir / name, "wb") as out:
                while written < part_bytes:
                    block = handle.read(min(CHUNK, part_bytes - written))
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    written += len(block)
            if written == 0:
                (out_dir / name).unlink()
                break
            parts.append({"asset": name, "size": written, "sha256": digest.hexdigest()})
            index += 1
            if written < part_bytes:
                break
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=str(ROOT / "models"), help="folder that holds onnx/ (default: models/)")
    parser.add_argument("--out", default=str(ROOT / "dist" / "release"))
    parser.add_argument("--name", default="clef-flash-onnx")
    parser.add_argument("--version", default="v0.0.0")
    parser.add_argument("--max-part-gib", type=float, default=1.9,
                        help="assets above this are split (GitHub's limit is 2 GiB)")
    parser.add_argument("--dry-run", action="store_true", help="only list the assets (names and sizes); read and write nothing")
    args = parser.parse_args()

    onnx_dir = pathlib.Path(args.models) / "onnx"
    missing = [n for n in WEIGHTS + FILES if not (onnx_dir / n).exists()]
    if missing:
        raise SystemExit("not under %s: %s (run export/prepare_model.py first)" % (onnx_dir, ", ".join(missing)))
    limit = int(args.max_part_gib * (1 << 30))
    names = WEIGHTS + FILES

    if args.dry_run:
        total = assets = 0
        for rel in names:
            size = (onnx_dir / rel).stat().st_size
            asset = rel.replace("/", "__")
            if size > limit:
                sizes = part_sizes(size, limit)
                for i, s in enumerate(sizes):
                    print("%-52s %10.1f MB" % ("%s.part%03d" % (asset, i), s / 1e6))
                assets += len(sizes)
            else:
                print("%-52s %10.1f MB" % (asset, size / 1e6))
                assets += 1
            total += size
        print("\n%d assets (+ manifest.json, SHA256SUMS), %.2f GB; the largest asset is limited to %.2f GiB (GitHub: < 2 GiB)"
              % (assets, total / 1e9, args.max_part_gib))
        return 0

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel in names:
        path = onnx_dir / rel
        asset = rel.replace("/", "__")
        size = path.stat().st_size
        entry = {"path": "onnx/" + rel, "asset": asset, "size": size, "sha256": sha256_of(path), "parts": None}
        if size > limit:
            entry["parts"] = split(path, out, asset, limit)
            print("split  %-44s %9.1f MB -> %d parts" % (rel, size / 1e6, len(entry["parts"])))
        else:
            link_or_copy(path, out / asset)
            print("asset  %-44s %9.1f MB" % (rel, size / 1e6))
        entries.append(entry)

    manifest = {"format": 1, "name": args.name, "version": args.version,
                "upstream": {"weights": "Cloudflare/clef-flash", "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
                             "graph": "ollaya-dev/clef", "graph_revision": "85b1c5cd90df6308ea95990c3e8a4f207454dad0"},
                "files": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "SHA256SUMS").write_text("".join("%s  %s\n" % (e["sha256"], e["path"]) for e in entries), encoding="utf-8")
    too_big = [a.name for a in out.iterdir() if a.stat().st_size >= (2 << 30)]
    if too_big:
        raise SystemExit("asset(s) of 2 GiB or more: %s" % ", ".join(too_big))
    print("wrote %s: %d files, %d assets" % (out, len(entries), len(list(out.iterdir()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
