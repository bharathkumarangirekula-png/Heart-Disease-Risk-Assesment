"""Patient logging helpers for the Heart Disease Risk Assessment Agent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils import LOGS_DIR, ensure_directories

LOG_FILE = LOGS_DIR / "patient_logs.csv"


class DatabaseError(RuntimeError):
    """Raised when patient log operations fail."""


class PatientDatabase:
    """Store prediction records in a CSV file."""

    def __init__(self) -> None:
        ensure_directories()
        self.log_file = LOG_FILE
        if not self.log_file.exists():
            self._initialize_file()

    def _initialize_file(self) -> None:
        """Create the patient log CSV with headers."""
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "patient_id",
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
                "prediction",
                "probability",
                "risk_level",
            ]
        )
        df.to_csv(self.log_file, index=False)

    def append_record(self, values: Dict[str, Any], result: Dict[str, Any], patient_id: str) -> None:
        """Append a patient prediction record to the CSV log."""
        try:
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient_id": patient_id,
                "age": values.get("age", ""),
                "sex": values.get("sex", ""),
                "cp": values.get("cp", ""),
                "trestbps": values.get("trestbps", ""),
                "chol": values.get("chol", ""),
                "fbs": values.get("fbs", ""),
                "restecg": values.get("restecg", ""),
                "thalach": values.get("thalach", ""),
                "exang": values.get("exang", ""),
                "oldpeak": values.get("oldpeak", ""),
                "slope": values.get("slope", ""),
                "ca": values.get("ca", ""),
                "thal": values.get("thal", ""),
                "prediction": result["prediction"],
                "probability": result["probability_percent"],
                "risk_level": result["risk_level"],
            }
            df = pd.read_csv(self.log_file)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            df.to_csv(self.log_file, index=False)
        except Exception as exc:  # pragma: no cover - defensive error handling
            raise DatabaseError(f"Unable to write patient log: {exc}") from exc

    def load_records(self) -> pd.DataFrame:
        """Return all patient records from the CSV log."""
        try:
            if not self.log_file.exists():
                self._initialize_file()
            return pd.read_csv(self.log_file)
        except Exception as exc:  # pragma: no cover - defensive error handling
            raise DatabaseError(f"Unable to read patient log: {exc}") from exc
