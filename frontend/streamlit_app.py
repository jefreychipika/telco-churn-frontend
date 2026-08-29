"""
streamlit_app.py
-----------------
Streamlit dashboard for the Telco Customer Churn app.

This file deliberately contains NO model logic and NO database logic -
its only job is to collect input from the user, send it to the FastAPI
backend, and display what comes back. Keeping the frontend "dumb" like
this means the model or database can change without touching this file
at all, as long as the API contract (schemas.py) stays the same.

Run with:
    streamlit run streamlit_app.py

Make sure the FastAPI backend (main.py) is already running first -
by default this expects it at http://localhost:8000
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Telco Customer Churn Predictor", page_icon="📉")
st.title("📉 Telco Customer Churn Predictor")
st.write(
    "Enter a customer's account details below to predict whether they are "
    "likely to churn (cancel their service)."
)

tab_predict, tab_history = st.tabs(["Predict", "Prediction History"])

# ---------------------------------------------------------------------------
# TAB 1: Prediction form
# ---------------------------------------------------------------------------
with tab_predict:
    with st.form("customer_form"):
        st.subheader("Customer details")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has a Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0
            )
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges)
            )

        with col2:
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )

        st.markdown("**Add-on services**")
        col3, col4 = st.columns(2)
        service_options = ["No", "Yes", "No internet service"]
        with col3:
            online_security = st.selectbox("Online Security", service_options)
            online_backup = st.selectbox("Online Backup", service_options)
            device_protection = st.selectbox("Device Protection", service_options)
        with col4:
            tech_support = st.selectbox("Tech Support", service_options)
            streaming_tv = st.selectbox("Streaming TV", service_options)
            streaming_movies = st.selectbox("Streaming Movies", service_options)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # This payload's keys must match backend/schemas.py's CustomerData
        payload = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            probability_pct = result["churn_probability"] * 100
            if result["prediction"] == 1:
                st.error(f"⚠️ {result['prediction_label']} — {probability_pct:.1f}% probability")
            else:
                st.success(f"✅ {result['prediction_label']} — {probability_pct:.1f}% probability")

            st.caption(f"Saved to database as record #{result['record_id']}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not reach the API at {API_URL}. "
                "Make sure the FastAPI backend is running (uvicorn main:app)."
            )
        except requests.exceptions.HTTPError as exc:
            st.error(f"The API returned an error: {exc}")


# ---------------------------------------------------------------------------
# TAB 2: Prediction history, pulled from MySQL via the API
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Recent predictions")
    limit = st.slider("Number of records to show", 5, 100, 20)

    if st.button("Refresh history"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/predictions", params={"limit": limit}, timeout=10)
        response.raise_for_status()
        history = response.json()

        if history:
            df_history = pd.DataFrame(history)
            df_history["churn_probability"] = (df_history["churn_probability"] * 100).round(1)
            df_history = df_history.rename(
                columns={
                    "churn_probability": "churn_probability_%",
                    "prediction": "will_churn",
                }
            )
            st.dataframe(df_history, use_container_width=True)
        else:
            st.info("No predictions have been made yet.")

    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not reach the API at {API_URL}. "
            "Make sure the FastAPI backend is running (uvicorn main:app)."
        )
