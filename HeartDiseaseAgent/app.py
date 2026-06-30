"""Streamlit application for heart disease risk assessment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from database import PatientDatabase, DatabaseError
from predictor import Predictor, ModelLoadError
from report_generator import build_report, save_report
from utils import FEATURE_NAMES, RISK_LEVELS, ensure_directories

st.set_page_config(page_title="Heart Disease Risk Assessment Agent", page_icon="❤️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_predictor() -> Predictor:
    """Load the model once and cache it across Streamlit sessions."""
    return Predictor()


@st.cache_data(show_spinner=False)
def load_logs() -> pd.DataFrame:
    """Load the patient log CSV and cache it."""
    db = PatientDatabase()
    return db.load_records()


def create_sidebar() -> None:
    """Create a sidebar navigation experience."""
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=90)
    st.sidebar.title("HeartCare AI")
    st.sidebar.caption("AI-powered heart disease risk monitoring")

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Prediction", "Patient Logs", "Analytics Dashboard", "About"],
        index=0,
    )
    st.session_state["page"] = page


def render_home() -> None:
    """Render the home page."""
    st.title("❤️ Heart Disease Risk Assessment Agent")
    st.markdown("A modern healthcare monitoring agent that estimates heart disease risk from clinical indicators.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("This application uses the trained Phase 1 model artifacts and provides a safe, explainable workflow for patient assessment.")
        st.subheader("Core features")
        st.markdown("- Predict heart disease risk from patient features")
        st.markdown("- Generate downloadable health reports")
        st.markdown("- Maintain patient monitoring logs")
        st.markdown("- Visualize analytics and trend patterns")
    with col2:
        st.success("Best practice: review all predictions with a licensed clinician and use them as supportive decision tools.")
        st.metric("Model Status", "Loaded")
        st.metric("Monitoring Logs", "CSV")


def render_prediction() -> None:
    """Render the prediction page."""
    st.title("🩺 Risk Prediction")
    st.caption("Enter the patient clinical details to generate a prediction and report.")

    with st.form("prediction_form"):
        st.subheader("Patient Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=220, value=120)
            chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar", [0, 1], format_func=lambda x: "False" if x == 0 else "True")
        with col2:
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.slider("Oldpeak", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
            slope = st.selectbox("ST Slope", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3], help="0 to 3")
        with col3:
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], help="0=normal, 1=fixed defect, 2=reversable defect, 3=unknown")
            st.markdown("### Instructions")
            st.caption("Ensure values reflect the most recent assessment before running prediction.")

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        values = {
            "age": int(age),
            "sex": int(sex),
            "cp": int(cp),
            "trestbps": int(trestbps),
            "chol": int(chol),
            "fbs": int(fbs),
            "restecg": int(restecg),
            "thalach": int(thalach),
            "exang": int(exang),
            "oldpeak": float(oldpeak),
            "slope": int(slope),
            "ca": int(ca),
            "thal": int(thal),
        }

        try:
            predictor = load_predictor()
            result = predictor.predict(values)
            patient_id = f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            db = PatientDatabase()
            db.append_record(values, result, patient_id)
            report_text = build_report(result, values, patient_id)
            report_path = save_report(report_text, patient_id)

            st.success("Prediction completed successfully.")
            st.metric("Patient ID", patient_id)
            st.metric("Prediction", result["prediction"])
            st.metric("Probability", f"{result['probability_percent']}%")
            st.metric("Risk Level", result["risk_level"])

            if result["risk_level"] == RISK_LEVELS["high"]:
                st.error("High risk detected. Please consult a healthcare professional promptly.")
            elif result["risk_level"] == RISK_LEVELS["moderate"]:
                st.warning("Moderate risk detected. Continue monitoring and follow lifestyle guidance.")
            else:
                st.success("Low risk detected. Continue preventive care and routine checkups.")

            with st.expander("View report"):
                st.text(report_text)
            with st.expander("Download report"):
                with open(report_path, "rb") as report_file:
                    st.download_button(
                        label="Download text report",
                        data=report_file,
                        file_name=report_path.name,
                        mime="text/plain",
                    )
        except ModelLoadError as exc:
            st.error(f"Unable to load the model: {exc}")
        except DatabaseError as exc:
            st.error(f"Unable to save the monitoring record: {exc}")
        except Exception as exc:  # pragma: no cover - defensive error handling
            st.error(f"Prediction failed: {exc}")


def render_logs() -> None:
    """Render the patient logs page."""
    st.title("📋 Patient Monitoring Logs")
    st.caption("Search and review prior patient assessments.")

    try:
        records = load_logs()
        if records.empty:
            st.info("No patient records available yet.")
            return

        search_term = st.text_input("Search by patient ID, prediction, or risk level")
        if search_term:
            mask = records.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
            records = records[mask]

        st.dataframe(records, use_container_width=True, hide_index=True)
    except DatabaseError as exc:
        st.error(f"Unable to load logs: {exc}")


def render_dashboard() -> None:
    """Render the analytics dashboard."""
    st.title("📊 Analytics Dashboard")

    try:
        records = load_logs()
        if records.empty:
            st.info("No analysis data available yet.")
            return

        total_patients = len(records)
        high_risk = int((records["risk_level"] == RISK_LEVELS["high"]).sum())
        low_risk = int((records["risk_level"] == RISK_LEVELS["low"]).sum())
        avg_prob = round(float(records["probability"].mean()), 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", total_patients)
        col2.metric("High Risk", high_risk)
        col3.metric("Low Risk", low_risk)
        col4.metric("Average Probability", f"{avg_prob}%")

        counts = records["risk_level"].value_counts().reindex([RISK_LEVELS["low"], RISK_LEVELS["moderate"], RISK_LEVELS["high"]], fill_value=0)
        st.bar_chart(counts.rename("Count"))

        prediction_counts = records["prediction"].value_counts()
        st.bar_chart(prediction_counts.rename("Count"))

        time_series = records.copy()
        time_series["timestamp"] = pd.to_datetime(time_series["timestamp"], errors="coerce")
        time_series = time_series.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not time_series.empty:
            st.line_chart(time_series.set_index("timestamp")["probability"].rename("Probability (%)"))
    except Exception as exc:  # pragma: no cover - defensive error handling
        st.error(f"Analytics failed: {exc}")


def render_about() -> None:
    """Render the about page."""
    st.title("ℹ️ About")
    st.markdown("This application demonstrates a practical healthcare AI workflow using a trained classification model and a modern Streamlit interface.")
    st.markdown("- Built with Python and Streamlit")
    st.markdown("- Uses the trained model artifact from Phase 1")
    st.markdown("- Designed for educational and prototype healthcare monitoring scenarios")


def main() -> None:
    """Application entrypoint."""
    ensure_directories()
    create_sidebar()
    page = st.session_state.get("page", "Home")

    if page == "Prediction":
        render_prediction()
    elif page == "Patient Logs":
        render_logs()
    elif page == "Analytics Dashboard":
        render_dashboard()
    elif page == "About":
        render_about()
    else:
        render_home()


if __name__ == "__main__":
    main()
