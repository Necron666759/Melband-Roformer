"""BS-RoFormer vocal separation adapter backed by pymss."""
from __future__ import annotations

from .pymss_adapter import PymssAdapter


class BSRoformerAdapter(PymssAdapter):
    BACKEND = "bs-roformer-pymss"
    MODEL_TYPE = "bs_roformer"
