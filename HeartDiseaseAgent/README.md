# Heart Disease Risk Assessment Agent

## Overview
The Heart Disease Risk Assessment Agent is a complete Python Streamlit application that loads a trained machine learning model and provides heart disease risk assessment for patient records. It supports prediction, report generation, patient monitoring logs, and analytics dashboards.

## Features
- Predict risk using a trained model artifact
- Generate downloadable health reports
- Save patient predictions in CSV logs
- Display patient monitoring history
- Visualize analytics with charts
- Modern Streamlit UI with sidebar navigation

## Project Structure
```text
HeartDiseaseAgent/
├── app.py
├── predictor.py
├── report_generator.py
├── database.py
├── utils.py
├── requirements.txt
├── models/
│   ├── heart_disease_model.pkl
│   └── scaler.pkl
├── logs/
│   └── patient_logs.csv
├── reports/
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Running the App
```bash
streamlit run app.py
```

## Screenshots
Add screenshots of the Home, Prediction, Logs, and Analytics pages here.

## Future Enhancements
- Add user authentication
- Support SQLite instead of CSV
- Improve explainability with SHAP
- Integrate hospital-style dashboards
