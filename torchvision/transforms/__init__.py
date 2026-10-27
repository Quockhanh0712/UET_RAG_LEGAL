"""
Minimal transforms package compatible with transformers image_utils import path.
"""

from enum import Enum


class InterpolationMode(Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest_exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    HAMMING = "hamming"
    BOX = "box"
