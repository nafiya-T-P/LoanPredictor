"""Shared settings and cleaning code, used by every step."""
import numpy as np
import pandas as pd

DATA_PATH = "data/loan_data.csv"

# Only the fields that appear in the Streamlit form
NUM_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
CREDIT_COL = ["Credit_History"]
CAT_COLS = ["Education", "Self_Employed", "Property_Area"]
FEATURES = NUM_COLS + CREDIT_COL + CAT_COLS
TARGET = "Loan_Status"


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Fix messy text labels: ' male ', 'FEMALE', 'Semi-Urban', 'not graduate' ...
    for col in ["Gender", "Education", "Property_Area", "Married", "Self_Employed", "Dependents"]:
        df[col] = (df[col].astype("string").str.strip().str.lower()
                   .str.replace("-", "", regex=False))
    df["Education"] = df["Education"].replace({"graduate": "Graduate", "not graduate": "Not Graduate"})
    df["Property_Area"] = df["Property_Area"].replace(
        {"urban": "Urban", "rural": "Rural", "semiurban": "Semiurban"})
    df["Self_Employed"] = df["Self_Employed"].replace({"yes": "Yes", "no": "No"})

    # 3. Impossible values become missing (a loan of 0 or less makes no sense)
    df.loc[df["LoanAmount"] <= 0, "LoanAmount"] = np.nan

    # 4. Target as 1 = Approved, 0 = Rejected
    df["Approved"] = (df[TARGET] == "Y").astype(int)

    # Object columns -> plain python objects so sklearn imputers are happy
    for col in CAT_COLS:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df