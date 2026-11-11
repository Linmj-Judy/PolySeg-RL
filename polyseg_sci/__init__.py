"""Dependency-free PolySeg-Sci scientific core.

The package contains schemas, segmentation, prediction, risk allocation,
continuation, accounting, and verification primitives.  It intentionally has
no dependency on verl, OpenAI4S, Ray, or a trainer registry.
"""

from .core import *  # noqa: F401,F403
