# -*- coding: utf-8 -*-
"""Gate: the numpy/Pillow image path equals transformers' own.

Run with `uv run --group export` (transformers 5.10.2). Compares, over many image shapes:

* pixel_values / image_grid_thw  vs  the processor from processor_config.json (what Cloudflare's code uses) after the same
  decode and 448-px cap;
* bilinear_indices / bilinear_weights  vs  transformers.vision_utils.get_vision_bilinear_indices_and_weights;
* vision position_ids  vs  get_vision_position_ids.
"""
import glob
import io
import pathlib
import random
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_decide.runtime import default_model_dir  # noqa: E402
from clef_decide.image import (ImageConfig, bilinear_inputs, decode_image, fit_long_side, prepare_images,  # noqa: E402
                                  vision_position_ids)


def main() -> int:
    from PIL import Image
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids

    from transformers import AutoProcessor
    cfg = ImageConfig.from_pretrained(default_model_dir())
    proc = AutoProcessor.from_pretrained(str(default_model_dir())).image_processor      # what Cloudflare's code uses
    # AutoProcessor may pick the torchvision backend, whose resize differs from Pillow's by up to one 8-bit level; like
    # Strands' own vision engine, the oracle is pinned to the PIL backend (export/make_golden_image.py does the same)
    from clef_decide.image import pil_image_processor
    proc = pil_image_processor(default_model_dir())
    rng = random.Random(1)
    images = []
    for w, h in [(448, 448), (640, 480), (480, 640), (1000, 300), (300, 1000), (97, 61), (33, 500), (2000, 1500), (449, 447), (224, 224), (5, 7)]:
        arr = np.frombuffer(bytes(rng.randrange(256) for _ in range(w * h * 3)), dtype=np.uint8).reshape(h, w, 3)
        images.append(("noise %dx%d" % (w, h), Image.fromarray(arr)))
    rgba = Image.new("RGBA", (300, 200), (255, 0, 0, 0))
    rgba.paste((0, 0, 255, 255), (50, 50, 150, 150))
    images.append(("rgba", rgba))
    for p in sorted(glob.glob(str(ROOT / "artifacts" / "images" / "*.jpg")))[:6]:
        images.append((pathlib.Path(p).name, Image.open(p)))

    bad = 0
    for name, img in images:
        mine = prepare_images([img], cfg)
        up = fit_long_side(decode_image(img), cfg.long_side)
        ref = proc(images=[up], return_tensors="pt")
        same_grid = np.array_equal(mine.image_grid_thw, ref["image_grid_thw"].numpy())
        dpx = float(np.abs(mine.pixel_values - ref["pixel_values"].numpy()).max()) if same_grid else float("inf")
        grid = torch.tensor(mine.image_grid_thw, dtype=torch.long)
        bi, bw = get_vision_bilinear_indices_and_weights(grid, num_grid_per_side=48, spatial_merge_size=2)
        my_bi, my_bw = bilinear_inputs(mine.image_grid_thw, 48, 2)
        pos = get_vision_position_ids(grid, 2)
        ok = (same_grid and dpx < 1e-6 and np.array_equal(my_bi, bi.numpy())
              and float(np.abs(my_bw - bw.numpy()).max()) < 1e-5
              and np.array_equal(vision_position_ids(mine.image_grid_thw, 2), pos.numpy()))
        bad += not ok
        print("%-22s grid %-12s max|d pixel| %-9s  %s" % (name, mine.image_grid_thw[0].tolist(), "%.1e" % dpx, "ok" if ok else "DIFFERS"))
    print("\nfailures: %d of %d" % (bad, len(images)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
