"""Step 2: preprocessing pipeline, model comparison, save the best model."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from common import load_and_clean, FEATURES, NUM_COLS, CREDIT_COL, CAT_COLS

SEED = 42
df = load_and_clean()
X, y = df[FEATURES], df["Approved"]

# Stratified split keeps the 75/25 approval ratio in both train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)


def make_preprocessor(scale):
    # Income and loan amount are very skewed, so log1p first
    num_steps = [("impute", SimpleImputer(strategy="median")),
                 ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        # Credit history: fill with most common value AND add a "was missing" flag
        ("credit", SimpleImputer(strategy="most_frequent", add_indicator=True), CREDIT_COL),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])


# CHANGED: no class_weight / scale_pos_weight. Class weighting pushes the predicted
# probabilities away from reality (the app said ~53% when ~75% were truly approved).
# Without it, the probability shown in the app matches the approval rate in the data.
models = {
    "Logistic Regression": (True, LogisticRegression(max_iter=1000)),
    "Decision Tree": (False, DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                                    random_state=SEED)),
    "Random Forest": (False, RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                    n_jobs=-1, random_state=SEED)),
}
try:  # XGBoost is optional
    from xgboost import XGBClassifier
    models["XGBoost"] = (False, XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                              eval_metric="logloss", random_state=SEED))
except ImportError:
    print("xgboost not installed, skipping it")

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
results, fitted = {}, {}
for name, (scale, clf) in models.items():
    pipe = Pipeline([("prep", make_preprocessor(scale)), ("model", clf)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        "cv_roc_auc": round(cv_auc, 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "precision_rejected": round(precision_score(y_test, pred, pos_label=0), 4),
        "recall_rejected": round(recall_score(y_test, pred, pos_label=0), 4),
        "f1_rejected": round(f1_score(y_test, pred, pos_label=0), 4),
        "precision_approved": round(precision_score(y_test, pred), 4),
        "recall_approved": round(recall_score(y_test, pred), 4),
        "f1_approved": round(f1_score(y_test, pred), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),  # [[TN,FP],[FN,TP]]
        "report": classification_report(y_test, pred, target_names=["Rejected", "Approved"]),
    }
    fitted[name] = pipe

table = pd.DataFrame(results).T[["cv_roc_auc", "test_roc_auc", "precision_rejected",
                                 "recall_rejected", "f1_rejected", "f1_approved"]]
print(table.to_string())

best = max(results, key=lambda n: results[n]["cv_roc_auc"])  # choose on CV, not on test
print("\nBest model (by cross-validated ROC-AUC):", best)
print(results[best]["report"])
print("Confusion matrix [[TN, FP],[FN, TP]]:", results[best]["confusion_matrix"])

# ---------------------------------------------------------------- feature importance
# One-hot columns are added together per original field, then scaled so the
# biggest feature = 100 (this is what the About page shows).
PRETTY = {"Credit_History": "Credit history", "Credit_Missing": "Credit missing",
          "Self_Employed": "Self employed", "Education": "Education",
          "Property_Area": "Property area", "ApplicantIncome": "Applicant income",
          "CoapplicantIncome": "Co-applicant income", "LoanAmount": "Loan amount",
          "Loan_Amount_Term": "Loan term"}


def source_feature(col_name):
    block, _, rest = col_name.partition("__")
    if block == "credit":
        return "Credit_Missing" if "missingindicator" in rest else "Credit_History"
    if block == "cat":
        return next(c for c in CAT_COLS if rest.startswith(c))
    return rest


pipe = fitted[best]
names = list(pipe.named_steps["prep"].get_feature_names_out())
model = pipe.named_steps["model"]
if hasattr(model, "feature_importances_"):
    imp = model.feature_importances_
elif hasattr(model, "coef_"):
    imp = np.abs(model.coef_[0])
else:
    imp = None

importance = {}
if imp is not None:
    grouped = {}
    for n, v in zip(names, imp):
        key = PRETTY[source_feature(n)]
        grouped[key] = grouped.get(key, 0.0) + float(v)
    top = max(grouped.values()) or 1.0
    importance = dict(sorted(((k, round(100 * v / top, 1)) for k, v in grouped.items()),
                             key=lambda kv: -kv[1]))

# "Typical applicant" values; the Predict page swaps each field to these
# one at a time to show what raised / lowered an estimate.
reference = {c: float(X_train[c].median()) for c in NUM_COLS}
reference["Credit_History"] = float(X_train["Credit_History"].mode().iloc[0])
for c in CAT_COLS:
    reference[c] = str(X_train[c].mode().iloc[0])

joblib.dump(pipe, "outputs/model.joblib")
with open("outputs/metrics.json", "w") as f:
    json.dump({"best_model": best, "results": results, "feature_importance": importance,
               "reference": reference, "n_rows": len(df),
               "n_train": len(X_train), "n_test": len(X_test)}, f, indent=2)
print("\nSaved outputs/model.joblib and outputs/metrics.json")