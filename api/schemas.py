"""Pydantic request / response models for the API."""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "Month": "January",
    "Age": 28,
    "Occupation": "Scientist",
    "Annual_Income": 19114.12,
    "Monthly_Inhand_Salary": 1824.84,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 4,
    "Interest_Rate": 3,
    "Num_of_Loan": 4,
    "Type_of_Loan": "Auto Loan, Credit-Builder Loan, Personal Loan, and Home Equity Loan",
    "Delay_from_due_date": 3,
    "Num_of_Delayed_Payment": 7,
    "Changed_Credit_Limit": 11.27,
    "Num_Credit_Inquiries": 4,
    "Credit_Mix": "Good",
    "Outstanding_Debt": 809.98,
    "Credit_Utilization_Ratio": 26.82,
    "Credit_History_Age": "22 Years and 9 Months",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 49.57,
    "Amount_invested_monthly": 80.42,
    "Payment_Behaviour": "High_spent_Small_value_payments",
    "Monthly_Balance": 312.49,
}

Num = Optional[Union[float, int, str]]


class CreditRecord(BaseModel):
    """One customer-month. Every field is optional; missing ones are imputed."""
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE}, extra="ignore")

    Month: Optional[str] = Field(None, description="Month name, e.g. 'January'")
    Age: Num = None
    Occupation: Optional[str] = None
    Annual_Income: Num = None
    Monthly_Inhand_Salary: Num = None
    Num_Bank_Accounts: Num = None
    Num_Credit_Card: Num = None
    Interest_Rate: Num = None
    Num_of_Loan: Num = None
    Type_of_Loan: Optional[str] = Field(None, description="Comma separated loan types")
    Delay_from_due_date: Num = None
    Num_of_Delayed_Payment: Num = None
    Changed_Credit_Limit: Num = None
    Num_Credit_Inquiries: Num = None
    Credit_Mix: Optional[str] = Field(None, description="Good / Standard / Bad")
    Outstanding_Debt: Num = None
    Credit_Utilization_Ratio: Num = None
    Credit_History_Age: Num = Field(None, description="'22 Years and 9 Months' or months as a number")
    Payment_of_Min_Amount: Optional[str] = Field(None, description="Yes / No")
    Total_EMI_per_month: Num = None
    Amount_invested_monthly: Num = None
    Payment_Behaviour: Optional[str] = None
    Monthly_Balance: Num = None


class Prediction(BaseModel):
    credit_score: str
    class_id: int
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    records: list[CreditRecord] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    predictions: list[Prediction]
