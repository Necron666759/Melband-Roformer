"""MDX23C vocal separation adapter backed by pymss."""
from __future__ import annotations

from .pymss_adapter import PymssAdapter


class MDX23CAdapter(PymssAdapter):
    BACKEND = "mdx23c-pymss"
    MODEL_TYPE = "mdx23c"
