"""Step 1: check data quality, explore approval rates, save outputs/insights.json
(the Data Insights page reads this file)."""
import json
import pandas as pd
from common import load_and_clean, DATA_PATH

raw = pd.read_csv(DATA_PATH)
n_dupes = int(raw.duplicated().sum())
print("Raw shape:", raw.shape)
print("Duplicate rows:", n_dupes)
print("\nMissing values (raw):\n", raw.isna().sum()[raw.isna().sum() > 0])

n_bad_loan = int((raw.drop_duplicates()["LoanAmount"] <= 0).sum())

df = load_and_clean()
print("\nShape after cleaning:", df.shape)
print("Overall approval rate:", round(df["Approved"].mean(), 3))

df["IncomeBand"] = pd.qcut(df["ApplicantIncome"], 4,
                           labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])


def rates(col):
    s = df.groupby(col, observed=True)["Approved"].mean()
    return {str(k): round(float(v), 4) for k, v in s.items()}


credit = {str(int(k)): v for k, v in
          df.groupby("Credit_History")["Approved"].mean().round(4).items()}

insights = {
    "n_rows": len(df),
    "n_duplicates": n_dupes,
    "n_invalid_loan": n_bad_loan,
    "approval_rate": round(float(df["Approved"].mean()), 4),
    "credit": credit,                       # {"0": rate, "1": rate}
    "income": rates("IncomeBand"),
    "education": rates("Education"),
    "area": rates("Property_Area"),
}
for name in ("credit", "income", "education", "area"):
    print(f"\nApproval rate by {name}:", insights[name])

with open("outputs/insights.json", "w") as f:
    json.dump(insights, f, indent=2)
print("\nSaved outputs/insights.json")