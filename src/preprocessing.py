"""
Tests for src/preprocessing.py.

Run from the project folder:  python -m pytest
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import preprocessing  # noqa: E402


def make_image(width, height, mode="RGB", colour=(200, 30, 30)):
    if mode == "L":
        colour = 100
    elif mode == "RGBA":
        colour = colour + (255,)
    return Image.new(mode, (width, height), colour)


def test_output_has_fixed_length_and_range():
    features = preprocessing.prepare_image(make_image(117, 1377))  # tallest shape in the data
    assert features.shape == (preprocessing.n_features(),)
    assert features.min() >= 0.0 and features.max() <= 1.0


def test_different_formats_give_same_length():
    for mode in ("RGB", "RGBA", "L", "P"):
        features = preprocessing.prepare_image(make_image(300, 200).convert(mode))
        assert features.shape == (preprocessing.n_features(),)


def test_bytes_and_image_give_identical_result():
    """The API (bytes) must prepare an image exactly like training (file/image)."""
    img = make_image(250, 400)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    from_bytes = preprocessing.load_image_bytes(buffer.getvalue())
    from_image = preprocessing.prepare_image(img)
    assert np.array_equal(from_bytes, from_image)


def test_padding_is_white():
    """A tall image is padded left and right with white (value 1.0)."""
    features = preprocessing.prepare_image(make_image(20, 200, colour=(0, 0, 0)))
    size = preprocessing.IMAGE_SIZE
    pixels = features.reshape(size, size, -1)
    assert np.allclose(pixels[:, 0], 1.0)       # left edge is padding
    assert pixels[size // 2, size // 2].max() < 0.1  # centre is the black image