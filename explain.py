"""Plain-language explanation for one prediction (used by the Predict page).

Idea: swap each field, one at a time, to a "typical applicant" value and see how the
probability moves. If the applicant's real value gives a higher probability than the
typical one, it raised the estimate; if lower, it lowered the estimate.
This shows how the model reacts. It is not a bank's reasoning.
"""
import pandas as pd
from common import FEATURES

SMALL = 0.02   # changes smaller than 2 percentage points are called "small effect"
MIN = 0.005    # changes smaller than 0.5 points are ignored

LABEL = {"ApplicantIncome": "Applicant income", "CoapplicantIncome": "Co-applicant income",
         "LoanAmount": "Loan amount", "Loan_Amount_Term": "Loan term",
         "Credit_History": "Credit history", "Education": "Education",
         "Self_Employed": "Self employed", "Property_Area": "Property area"}


def _message(feature, row, helped):
    v = row[feature].iloc[0]
    if feature == "Credit_History":
        return "Credit history meets guidelines" if helped else "Credit history does not meet guidelines"
    if feature in ("ApplicantIncome", "CoapplicantIncome", "LoanAmount"):
        return f"{LABEL[feature]}: {v:,.0f}"
    if feature == "Loan_Amount_Term":
        return f"Loan term: {int(v)} months"
    return f"{LABEL[feature]}: {v}"


def explain(model, row, reference, p):
    """Returns (raised_list, lowered_list, what_if_sentence)."""
    alts = []
    for f in FEATURES:
        alt = row.copy()
        # Credit history is compared with the opposite value (good vs bad),
        # every other field with the typical applicant.
        alt[f] = (1.0 - float(row[f].iloc[0])) if f == "Credit_History" else reference[f]
        alts.append(alt)
    probs = model.predict_proba(pd.concat(alts, ignore_index=True))[:, 1]
    deltas = {f: p - float(q) for f, q in zip(FEATURES, probs)}

    def pick(sign):
        items = sorted((f for f, d in deltas.items() if sign * d > MIN),
                       key=lambda f: -abs(deltas[f]))[:2]
        out = []
        for f in items:
            text = _message(f, row, helped=(sign > 0))
            if abs(deltas[f]) < SMALL:
                text = f"{LABEL[f]}: small effect"
            out.append(text)
        return out or ["Nothing stands out"]

    credit = float(row["Credit_History"].iloc[0])
    flipped = row.copy()
    flipped["Credit_History"] = 0.0 if credit == 1.0 else 1.0
    p2 = float(model.predict_proba(flipped)[0, 1])
    what_if = (f"With credit history {int(flipped['Credit_History'].iloc[0])}, the estimate "
               f"{'drops' if p2 < p else 'rises'} from {p:.0%} to about {p2:.0%}.")
    return pick(+1), pick(-1), what_if