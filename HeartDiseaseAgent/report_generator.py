"""Generate downloadable health reports for the Heart Disease Risk Assessment Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from utils import REPORTS_DIR, sanitize_text


def build_report(result: Dict[str, Any], values: Dict[str, Any], patient_id: str) -> str:
    """Create a professional text report for a prediction."""
    lines = [
        "Heart Disease Risk Assessment Report",
        "===================================",
        f"Patient ID: {patient_id}",
        f"Prediction: {result['prediction']}",
        f"Probability: {result['probability_percent']}%",
        f"Risk Level: {result['risk_level']}",
        "",
        "Patient Information",
        "-------------------",
        f"Age: {values.get('age', '')}",
        f"Sex: {sanitize_text(values.get('sex', ''))}",
        f"Chest Pain Type: {sanitize_text(values.get('cp', ''))}",
        f"Resting Blood Pressure: {values.get('trestbps', '')}",
        f"Cholesterol: {values.get('chol', '')}",
        f"Fasting Blood Sugar: {sanitize_text(values.get('fbs', ''))}",
        "",
        "Assessment Summary",
        "------------------",
        "This report summarizes the predicted heart disease risk based on the provided clinical indicators.",
        "",
        "Lifestyle Recommendations",
        "-------------------------",
        "- Maintain a balanced diet rich in vegetables, fruits, legumes, and whole grains.",
        "- Engage in regular physical activity and follow a physician-approved exercise plan.",
        "- Keep body weight within a healthy range and monitor waist circumference.",
        "- Check blood pressure regularly and follow treatment guidance if needed.",
        "- Manage cholesterol through diet, exercise, and medical care.",
        "- Avoid smoking and minimize secondhand smoke exposure.",
        "- Practice stress management techniques such as mindfulness and relaxation.",
        "- Schedule regular medical checkups and discuss any new symptoms with a clinician.",
        "",
        "Clinical Note",
        "-------------",
        "This report is intended for informational use and should not replace medical advice.",
    ]
    return "\n".join(lines)


def save_report(report_text: str, patient_id: str) -> Path:
    """Persist a report to disk and return its path."""
    report_path = REPORTS_DIR / f"{patient_id}_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path
