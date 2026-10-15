"""Data cleaning for the raw bank / credit records.

The raw file is deliberately messy: numbers carry trailing underscores, placeholders such
as ``_``, ``NM`` or ``!@9#%8`` stand in for missing values, and many numeric columns hold
impossible values (age = -500, 5 000 credit cards, ...).  Every step below turns such
cells into NaN first and fills them afterwards - preferably with the customer's own other
months, otherwise with the median inside the model pipeline.

The same ``clean`` function is used for training, for the test file and by the API.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

MONTH_NUM = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}

_HIST_RE = re.compile(r"(\d+)\s*years?\s*(?:and)?\s*(\d+)\s*months?", re.I)

PLACEHOLDERS = {"_", "__", "_______", "NM", "!@9#%8", "#F%$D@*&8", "nan", "NA", ""}

CONST_NUM = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
             "Num_Credit_Card", "Interest_Rate", "Num_of_Loan"]
VAR_NUM = ["Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
           "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance",
           "Total_EMI_per_month"]
CAT_FILL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]


def _to_number(series: pd.Series) -> pd.Series:
    """'28_' -> 28.0, '_' -> NaN, junk -> NaN."""
    s = series.astype("string").str.replace("_", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def parse_history_age(value) -> float:
    """'22 Years and 9 Months' -> 273 (months). Plain numbers are taken as months."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    m = _HIST_RE.search(str(value))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def _clean_categorical(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.where(~s.isin(PLACEHOLDERS), pd.NA)


def _group_mode(df: pd.DataFrame, col: str) -> pd.Series:
    mode = df.groupby(C.GROUP_COL)[col].agg(lambda x: x.mode().iloc[0] if x.notna().any() else pd.NA)
    return df[C.GROUP_COL].map(mode)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy. Works with or without ``Customer_ID`` / ``Credit_Score``."""
    df = raw.copy()

    # make sure every expected column exists (API requests may omit fields)
    for col in C.NUMERIC_RAW + C.CATEGORICAL + ["Type_of_Loan", "Credit_History_Age", "Month"]:
        if col not in df.columns:
            df[col] = np.nan

    # 1) numeric columns: strip noise, convert, null out impossible values
    for col in C.NUMERIC_RAW:
        df[col] = _to_number(df[col]).astype("float64")
    for col, (lo, hi) in C.VALID_RANGE.items():
        df.loc[~df[col].between(lo, hi), col] = np.nan

    # 2) categorical placeholders -> missing
    for col in C.CATEGORICAL:
        df[col] = _clean_categorical(df[col])

    # 3) month number and credit history age in months
    df["Month_Num"] = df["Month"].map(MONTH_NUM).astype("float64")
    df["Credit_History_Age"] = df["Credit_History_Age"].map(parse_history_age).astype("float64")

    # 4) type of loan -> text, one flag per loan type (done before group filling)
    loans = df["Type_of_Loan"].astype("string").fillna("No Loan")
    for lt in C.LOAN_TYPES:
        df[f"Loan_{lt.replace(' ', '_').replace('-', '_')}"] = loans.str.contains(lt, regex=False).astype(int)

    # 5) fill gaps using the same customer's other months (training / test files only)
    if C.GROUP_COL in df.columns and df[C.GROUP_COL].notna().any():
        g = df.groupby(C.GROUP_COL)
        for col in CONST_NUM + VAR_NUM:
            df[col] = df[col].fillna(g[col].transform("median"))
        for col in CAT_FILL:
            df[col] = df[col].fillna(_group_mode(df, col))
        # history age grows by exactly one month per month -> align to a common base
        base = (df["Credit_History_Age"] - df["Month_Num"]).groupby(df[C.GROUP_COL]).transform("median")
        df["Credit_History_Age"] = df["Credit_History_Age"].fillna(base + df["Month_Num"])

    return df


def build_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features and return only the columns the model uses."""
    df = cleaned
    df["Debt_to_Income"] = df["Outstanding_Debt"] / (df["Annual_Income"] + 1)
    df["EMI_to_Salary"] = df["Total_EMI_per_month"] / (df["Monthly_Inhand_Salary"] + 1)
    df["Balance_to_Salary"] = df["Monthly_Balance"] / (df["Monthly_Inhand_Salary"] + 1)
    df["Invest_to_Salary"] = df["Amount_invested_monthly"] / (df["Monthly_Inhand_Salary"] + 1)
    loan_cols = [c for c in df.columns if c.startswith("Loan_")]
    df["Num_Loan_Types"] = df[loan_cols].sum(axis=1)

    for col in C.CATEGORICAL:                       # plain object dtype for sklearn
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df[feature_columns()]


def feature_columns() -> list[str]:
    loan_cols = [f"Loan_{lt.replace(' ', '_').replace('-', '_')}" for lt in C.LOAN_TYPES]
    numeric = [c for c in C.NUMERIC_RAW] + ["Credit_History_Age", "Month_Num"]
    engineered = ["Debt_to_Income", "EMI_to_Salary", "Balance_to_Salary",
                  "Invest_to_Salary", "Num_Loan_Types"]
    return numeric + C.CATEGORICAL + loan_cols + engineered


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """raw records -> model-ready feature frame."""
    return build_features(clean(raw))
