"""Utility helpers for the Heart Disease Risk Assessment Agent."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

RISK_LEVELS = {
    "low": "Low Risk",
    "moderate": "Moderate Risk",
    "high": "High Risk",
}


def ensure_directories() -> None:
    """Create required folders if they do not exist."""
    for directory in (MODELS_DIR, LOGS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_feature_dataframe(values: Dict[str, Any]) -> pd.DataFrame:
    """Convert user inputs into a DataFrame matching the model feature order."""
    row = {feature: values.get(feature, 0) for feature in FEATURE_NAMES}
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def to_json(data: Any) -> str:
    """Serialize Python data to JSON for storage or display."""
    return json.dumps(data, indent=2, default=str)


def sanitize_text(value: Any) -> str:
    """Return a human-friendly string for UI display."""
    if value is None:
        return ""
    return str(value)


def risk_level_from_probability(probability: float) -> str:
    """Map a prediction probability to a human-readable risk category."""
    if probability < 0.4:
        return RISK_LEVELS["low"]
    if probability < 0.7:
        return RISK_LEVELS["moderate"]
    return RISK_LEVELS["high"]
