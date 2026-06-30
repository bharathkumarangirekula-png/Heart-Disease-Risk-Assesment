"""Prediction utilities for the Heart Disease Risk Assessment Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from utils import MODELS_DIR, build_feature_dataframe, risk_level_from_probability


def resolve_artifact_path(filename: str) -> Path:
    """Locate a required artifact in the project folder or the workspace root."""
    candidates: List[Path] = []
    base_dir = Path(__file__).resolve().parent
    candidates.append(base_dir / "models" / filename)
    candidates.append(base_dir / filename)
    candidates.append(base_dir.parent / filename)
    candidates.append(base_dir.parent / "models" / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


MODEL_PATH = resolve_artifact_path("heart_disease_model.pkl")
SCALER_PATH = resolve_artifact_path("scaler.pkl")


class ModelLoadError(RuntimeError):
    """Raised when the model or scaler cannot be loaded."""


class Predictor:
    """Load the trained model once and expose prediction helpers."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and scaler from disk using joblib."""
        if not MODEL_PATH.exists():
            raise ModelLoadError(f"Model file not found at {MODEL_PATH}")

        self.model = joblib.load(MODEL_PATH)
        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
        else:
            self.scaler = None

    def predict(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Predict heart disease risk for a new patient record."""
        frame = build_feature_dataframe(values)

        if self.scaler is not None:
            scaled = self.scaler.transform(frame)
            probabilities = self.model.predict_proba(scaled)[0]
            prediction = int(self.model.predict(scaled)[0])
        else:
            probabilities = self.model.predict_proba(frame)[0]
            prediction = int(self.model.predict(frame)[0])

        probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        risk_level = risk_level_from_probability(probability)

        return {
            "prediction": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability": probability,
            "probability_percent": round(probability * 100, 2),
            "risk_level": risk_level,
            "features": frame.to_dict(orient="records")[0],
        }
