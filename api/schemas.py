"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Input features for churn prediction — all training features."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
            }
        }
    )

    # Categorical features
    gender: str = Field(..., description="Male | Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes | No")
    Dependents: str = Field(..., description="Yes | No")
    PhoneService: str = Field(..., description="Yes | No")
    MultipleLines: str = Field(..., description="Yes | No | No phone service")
    InternetService: str = Field(..., description="DSL | Fiber optic | No")
    OnlineSecurity: str = Field(..., description="Yes | No | No internet service")
    OnlineBackup: str = Field(..., description="Yes | No | No internet service")
    DeviceProtection: str = Field(..., description="Yes | No | No internet service")
    TechSupport: str = Field(..., description="Yes | No | No internet service")
    StreamingTV: str = Field(..., description="Yes | No | No internet service")
    StreamingMovies: str = Field(..., description="Yes | No | No internet service")
    Contract: str = Field(..., description="Month-to-month | One year | Two year")
    PaperlessBilling: str = Field(..., description="Yes | No")
    PaymentMethod: str = Field(
        ..., description="Electronic check | Mailed check | Bank transfer | Credit card"
    )

    # Numeric features
    tenure: float = Field(..., ge=0, description="Months as a customer")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly bill amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")


class PredictionResponse(BaseModel):
    """Churn prediction output."""

    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_level: str  # low | medium | high
    model_version: str
