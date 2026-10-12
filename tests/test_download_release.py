# -*- coding: utf-8 -*-
"""download_model.py (repository choice, file selection, a full round trip from a local release) and release/make_release.py."""
import argparse
import functools
import hashlib
import http.server
import json
import pathlib
import threading

import download_model as dl
from conftest import load_script


def _args(**kw):
    base = dict(repo=None, tag="latest", base_url=None, include=None, exclude=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_repository_priority_argument_then_environment_then_default(monkeypatch):
    monkeypatch.delenv("CLEF_ONNX_REPO", raising=False)
    assert dl.DEFAULT_REPO == "Kazuhito00/Clef-Flash-ONNX-CPU"
    assert dl.base_url(_args()) == "https://github.com/Kazuhito00/Clef-Flash-ONNX-CPU/releases/latest/download"
    monkeypatch.setenv("CLEF_ONNX_REPO", "someone/fork")
    assert dl.base_url(_args()) == "https://github.com/someone/fork/releases/latest/download"
    assert dl.base_url(_args(repo="other/repo", tag="v0.0.0")) == "https://github.com/other/repo/releases/download/v0.0.0"
    assert dl.base_url(_args(base_url="http://127.0.0.1:8000/")) == "http://127.0.0.1:8000"


def _entries():
    names = ["onnx/model-fp32.onnx", "onnx/model-fp32-vl.onnx", "onnx/vision_encoder.onnx", "onnx/vision_encoder.onnx.data",
             "onnx/model-00001-of-00004.safetensors", "onnx/tokenizer.json"]
    return [{"path": n, "asset": n.replace("/", "__"), "size": 1, "sha256": "0", "parts": None} for n in names]


def test_default_selection_is_everything():
    assert len(dl.select(_entries(), _args())) == 6


def test_text_only_excludes_the_vision_tower():
    got = [f["path"] for f in dl.select(_entries(), _args(exclude=["*vision_encoder*", "*-vl.onnx"]))]
    assert got == ["onnx/model-fp32.onnx", "onnx/model-00001-of-00004.safetensors", "onnx/tokenizer.json"]


def test_include_globs():
    got = [f["path"] for f in dl.select(_entries(), _args(include=["*.json"]))]
    assert got == ["onnx/tokenizer.json"]


def _fake_models(root: pathlib.Path):
    """Every file make_release.py wants, tiny; one fake weight file is big enough to be split."""
    mk = load_script("make_release", "release/make_release.py")
    onnx = root / "onnx"
    (onnx / "flash").mkdir(parents=True)
    pool = bytes(range(256)) * 40                                         # 10240 bytes
    for name in mk.WEIGHTS + mk.FILES:
        (onnx / name).write_bytes(pool if name == mk.WEIGHTS[0] else name.encode())
    return pool


def test_make_release_dry_run_lists_parts_without_writing(tmp_path, monkeypatch, capsys):
    mk = load_script("make_release", "release/make_release.py")
    _fake_models(tmp_path / "models")
    monkeypatch.setattr("sys.argv", ["make_release.py", "--models", str(tmp_path / "models"), "--dry-run",
                                     "--max-part-gib", str(4000 / (1 << 30))])
    assert mk.main() == 0
    out = capsys.readouterr().out
    assert "model-00001-of-00004.safetensors.part002" in out and "flash__decision.json" in out
    assert not (tmp_path / "dist").exists()


def test_download_round_trip_from_a_local_release(tmp_path, monkeypatch):
    mk = load_script("make_release", "release/make_release.py")
    pool = _fake_models(tmp_path / "models")
    monkeypatch.setattr("sys.argv", ["make_release.py", "--models", str(tmp_path / "models"), "--out", str(tmp_path / "dist"),
                                     "--max-part-gib", str(4000 / (1 << 30))])
    assert mk.main() == 0
    manifest = json.loads((tmp_path / "dist" / "manifest.json").read_text(encoding="utf-8"))
    weights = {f["path"]: f for f in manifest["files"]}["onnx/model-00001-of-00004.safetensors"]
    assert [p["size"] for p in weights["parts"]] == [4000, 4000, 2240] and weights["sha256"] == hashlib.sha256(pool).hexdigest()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "dist"))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d" % server.server_address[1]
        out = tmp_path / "got"
        monkeypatch.setattr("sys.argv", ["download_model.py", "--base-url", url, "--dest", str(out), "--exclude", "*vision_encoder*"])
        assert dl.main() == 0
        assert (out / "onnx" / "model-00001-of-00004.safetensors").read_bytes() == pool      # three parts joined, sha256 checked
        assert (out / "onnx" / "flash" / "decision.json").exists() and not (out / "onnx" / "vision_encoder.onnx").exists()
        assert not list(out.rglob("*.part")) and not list(out.rglob("*.dl"))
        monkeypatch.setattr("sys.argv", ["download_model.py", "--base-url", url, "--dest", str(out)])
        assert dl.main() == 0                                                                # a second run adds the rest
        assert (out / "onnx" / "vision_encoder.onnx").exists()
        assert len(manifest["files"]) == len([p for p in out.rglob("*") if p.is_file()])
    finally:
        server.shutdown()
