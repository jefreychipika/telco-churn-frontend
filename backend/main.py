"""
main.py
--------
FastAPI backend for the Telco Customer Churn app.

Responsibilities:
    1. Serve predictions from the trained model (POST /predict)
    2. Persist every prediction to MySQL, so there's a history to review
    3. Serve that history back to the dashboard (GET /predictions)

Run locally with:
    uvicorn main:app --reload --port 8000

Interactive API docs are then available at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import database
from predictor import predict
from schemas import CustomerData, PredictionHistoryItem, PredictionResponse

app = FastAPI(
    title="Telco Customer Churn API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different port) to call this
# API from the browser. In production, replace "*" with the exact
# frontend URL for tighter security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the API is running - useful for
    Streamlit to check connectivity before showing the prediction form.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """Score one customer and store the result in MySQL.

    Steps:
        1. Run the customer's data through the trained pipeline.
        2. Save the input + result as one row in `churn_predictions`.
        3. Return the prediction, probability, and the new row's id.
    """
    try:
        prediction, probability = predict(customer)
    except Exception as exc:
        # Anything from a malformed pipeline file to a bad input value
        # lands here - surfaced as a clean 500 instead of a raw traceback.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Build the row to insert, matching schema.sql's snake_case columns.
    record = {
        "gender": customer.gender,
        "senior_citizen": customer.SeniorCitizen,
        "partner": customer.Partner,
        "dependents": customer.Dependents,
        "tenure": customer.tenure,
        "monthly_charges": customer.MonthlyCharges,
        "total_charges": customer.TotalCharges,
        "contract": customer.Contract,
        "paperless_billing": customer.PaperlessBilling,
        "payment_method": customer.PaymentMethod,
        "phone_service": customer.PhoneService,
        "multiple_lines": customer.MultipleLines,
        "internet_service": customer.InternetService,
        "online_security": customer.OnlineSecurity,
        "online_backup": customer.OnlineBackup,
        "device_protection": customer.DeviceProtection,
        "tech_support": customer.TechSupport,
        "streaming_tv": customer.StreamingTV,
        "streaming_movies": customer.StreamingMovies,
        "prediction": prediction,
        "churn_probability": round(probability, 4),
    }

    try:
        record_id = database.save_prediction(record)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save to database: {exc}")

    label = "High churn risk" if prediction == 1 else "Low churn risk"

    return PredictionResponse(
        prediction=prediction,
        prediction_label=label,
        churn_probability=round(probability, 4),
        record_id=record_id,
    )


@app.get("/predictions", response_model=list[PredictionHistoryItem])
def get_predictions(limit: int = 20):
    """Return the most recent predictions, for the dashboard's history tab."""
    try:
        rows = database.get_recent_predictions(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read database: {exc}")

    # created_at comes back as a datetime object from MySQL; convert to
    # string so it matches the PredictionHistoryItem schema cleanly.
    for row in rows:
        row["created_at"] = str(row["created_at"])

    return rows
