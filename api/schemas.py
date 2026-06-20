"""
Pydantic schemas for request/response validation.
Full implementation in Phase 3.
"""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""

    tenure: int = Field(..., ge=0, description="Months as a customer")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly bill amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")
    Contract: str = Field(..., description="Month-to-month | One year | Two year")
    InternetService: str = Field(..., description="DSL | Fiber optic | No")
    PaymentMethod: str = Field(..., description="Payment method")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
            }
        }


class PredictionResponse(BaseModel):
    """Churn prediction output."""

    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_level: str  # low | medium | high
    model_version: str
