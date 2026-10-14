"""Data passed from the worker to the GUI, without exposing the worker's model.

Type-only imports keep presentation code independent of PyTorch at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from src.inference import Prediction


@dataclass(frozen=True)
class ModelReady:
    checkpoint_sha256: str
    load_seconds: float
    device: str
    model_loads: int
    predictions: int


@dataclass(frozen=True)
class AnalysisResult:
    request_id: int
    prediction: Prediction
    overlay: Image.Image | None
    warning: str
    prediction_seconds: float
    explanation_seconds: float
    model_loads: int
    predictions: int
