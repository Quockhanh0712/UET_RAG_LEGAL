"""
Lightweight torchvision stub for text-only workloads.

This project does not use torchvision features, but some transformer imports
touch vision utilities. A local stub avoids runtime crashes when the global
torch/torchvision installation is incompatible.
"""

from .transforms import InterpolationMode
from . import io

__all__ = ["InterpolationMode", "io"]
