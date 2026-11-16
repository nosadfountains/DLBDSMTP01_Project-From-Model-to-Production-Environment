"""
Shared image preprocessing for training (WP4) and the API (WP5).

Every image goes through exactly the same steps, wherever it comes from:

  1. turn it upright (some photos store their rotation separately)
  2. convert to grayscale             -> colour is not what defines the class
  3. pad to a square with white       -> keeps the shape, nothing is cut off
  4. resize to IMAGE_SIZE x IMAGE_SIZE pixels
  5. scale pixel values from 0-255 to 0.0-1.0 and flatten into one row

Why one shared file: if training and the API prepared images differently,
the model would give wrong answers without any error message.

The settings below are saved next to the trained model, so the API can
check that it uses the same ones.
"""

import io

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 64        # final width and height in pixels
GRAYSCALE = True       # True: 1 value per pixel; False: 3 values (red, green, blue)
PAD_COLOUR = 255       # white; most product photos have a light background


def settings():
    """Return the preprocessing settings as a dictionary (saved with the model)."""
    return {
        "image_size": IMAGE_SIZE,
        "grayscale": GRAYSCALE,
        "resize_method": "pad_to_square",
        "pad_colour": PAD_COLOUR,
        "scale": "0-1",
    }


def n_features():
    """Number of values per image after preprocessing (64 x 64 = 4,096 for grayscale)."""
    channels = 1 if GRAYSCALE else 3
    return IMAGE_SIZE * IMAGE_SIZE * channels


def prepare_image(img):
    """Turn one PIL image into a 1-D numpy array of floats between 0 and 1."""
    img = ImageOps.exif_transpose(img)               # 1. upright
    img = img.convert("L" if GRAYSCALE else "RGB")   # 2. grayscale (or plain RGB)

    width, height = img.size                         # 3. pad to a square
    side = max(width, height)
    fill = PAD_COLOUR if GRAYSCALE else (PAD_COLOUR,) * 3
    square = Image.new(img.mode, (side, side), fill)
    square.paste(img, ((side - width) // 2, (side - height) // 2))

    small = square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)  # 4.
    pixels = np.asarray(small, dtype=np.float32) / 255.0                       # 5.
    return pixels.reshape(-1)


def load_image_file(path):
    """Open an image file from disk and prepare it (used by training and the batch job)."""
    with Image.open(path) as img:
        return prepare_image(img)


def load_image_bytes(data):
    """Prepare an image that arrives as raw bytes, e.g. in an API request (WP5)."""
    with Image.open(io.BytesIO(data)) as img:
        return prepare_image(img)