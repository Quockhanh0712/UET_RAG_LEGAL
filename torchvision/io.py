"""
Minimal torchvision.io stub compatible with transformers image_utils.
"""

from enum import Enum


class ImageReadMode(Enum):
    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


def decode_image(*args, **kwargs):
    raise NotImplementedError("torchvision.io.decode_image is not available in this text-only stub.")


def read_video(*args, **kwargs):
    raise NotImplementedError("torchvision.io.read_video is not available in this text-only stub.")
