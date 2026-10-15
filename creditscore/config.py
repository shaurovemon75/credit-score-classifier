"""Central configuration: paths, label mapping and column groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.csv.gz"
TEST_PATH = DATA_DIR / "test.csv.gz"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_score_model.joblib"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
SUBMISSION_PATH = REPORT_DIR / "test_predictions.csv"
METRICS_PATH = REPORT_DIR / "metrics.json"

TARGET = "Credit_Score"
LABELS = ["Poor", "Standard", "Good"]            # class id = position in this list
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

RANDOM_STATE = 42

# ---- raw column groups ---------------------------------------------------------
DROP_COLS = ["ID", "Name", "SSN"]                 # identifiers, no predictive value
GROUP_COL = "Customer_ID"                         # used for imputation + leakage-free CV

DIRTY_NUMERIC = [                                 # numbers stored as text with '_' noise
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
    "Monthly_Balance",
]
NUMERIC_RAW = DIRTY_NUMERIC + [
    "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Delay_from_due_date", "Num_Credit_Inquiries", "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
]
CATEGORICAL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

# plausible ranges - anything outside is a data-entry error and becomes NaN
VALID_RANGE = {
    "Age": (14, 100),
    "Annual_Income": (0, 300_000),
    "Monthly_Inhand_Salary": (0, 25_000),
    "Num_Bank_Accounts": (0, 15),
    "Num_Credit_Card": (0, 15),
    "Interest_Rate": (1, 40),
    "Num_of_Loan": (0, 9),
    "Num_of_Delayed_Payment": (0, 28),
    "Changed_Credit_Limit": (-10, 40),
    "Num_Credit_Inquiries": (0, 20),
    "Outstanding_Debt": (0, 6_000),
    "Total_EMI_per_month": (0, 1_500),
    "Amount_invested_monthly": (0, 2_000),
    "Monthly_Balance": (-100, 2_000),
}

LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
    "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan",
]
