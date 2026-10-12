# -*- coding: utf-8 -*-
"""Gate: with no image, the VL graph (model-fp32-vl.onnx) answers every golden text case like Cloudflare's code.

The VL graph is the text graph plus three inputs (image_embeds, image_positions, mrope_positions); with N = 0 and
mrope_positions = arange repeated three times it must compute the same thing. This is verify.py run through that graph.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.run([sys.executable, str(ROOT / "verify" / "verify.py"), "--vl", *sys.argv[1:]]).returncode)
