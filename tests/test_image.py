# -*- coding: utf-8 -*-
"""Image preprocessing arithmetic (no model): the resize grid, the placeholder text and the M-RoPE positions."""
import numpy as np

from clef_decide import image


def test_smart_resize_is_a_multiple_of_the_factor():
    h, w = image.smart_resize(300, 500, 32, 65536, 448 * 448)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 448 * 448


def test_image_prefix_has_one_pad_per_merged_patch():
    text = image.image_prefix_text([6, 4])
    assert text.count("<|image_pad|>") == 10 and text.count("<|vision_start|>") == 2 and text.endswith("\n")


def test_positions_without_images_count_up_on_all_three_axes():
    pos = image.position_ids([1, 2, 3, 4], np.zeros((0, 3), dtype=np.int64))
    assert pos.shape == (3, 4) and (pos == np.arange(4)).all()
