"""
schemas.py
-----------
Pydantic models that define exactly what the API expects to receive and
what it sends back. FastAPI uses these to validate incoming requests
automatically (e.g. rejecting a request if `tenure` is missing or isn't a
number) and to generate the interactive docs at /docs.

Field names here deliberately match the original Telco dataset's column
names (e.g. `MonthlyCharges`, not `monthly_charges`) so that a dataframe
built straight from a CustomerData object can be fed directly into the
trained pipeline without any renaming.
"""

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """One customer's details, as entered on the Streamlit form."""

    gender: str = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])

    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])

    MonthlyCharges: float = Field(..., ge=0, description="USD per month")
    TotalCharges: float = Field(..., ge=0, description="USD, lifetime total")


class PredictionResponse(BaseModel):
    """What the API sends back after scoring a customer."""

    prediction: int  # 1 = will churn, 0 = will stay
    prediction_label: str  # human-readable version, e.g. "High churn risk"
    churn_probability: float  # 0.0 - 1.0
    record_id: int  # id of the row saved in MySQL


class PredictionHistoryItem(BaseModel):
    """One row from the churn_predictions table, for the history view."""

    id: int
    gender: str
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    prediction: int
    churn_probability: float
    created_at: str
