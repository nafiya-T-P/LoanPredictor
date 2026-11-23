"""Streamlit app: Home / Predict / Data Insights / About the Model.
Run with:  python -m streamlit run app.py
"""
import json
from html import escape
import joblib
import pandas as pd
import streamlit as st
from common import FEATURES
import charts
from charts import bar_rows, card, stat, grid
from explain import explain

st.set_page_config(page_title="Loan Predictor", page_icon="🏦", layout="wide")

with open("styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ data loading
@st.cache_resource
def load_model():
    return joblib.load("outputs/model.joblib")


@st.cache_data
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


NOTE = ('<div class="note"><b>Please note:</b> this is a learning project trained on a sample '
        'dataset. Results are only model estimates, not real lending decisions.</div>')

CREDIT_OPTS = ["1 – Meets guidelines", "0 – Does not meet guidelines"]
EXAMPLES = {
    "Strong applicant": dict(app_inc=60000, co_inc=15000, loan=120, term=360,
                             credit=CREDIT_OPTS[0], edu="Graduate", emp="No", area="Semiurban"),
    "Borderline applicant": dict(app_inc=25000, co_inc=0, loan=200, term=240,
                                 credit=CREDIT_OPTS[0], edu="Not Graduate", emp="Yes", area="Rural"),
    "No credit history": dict(app_inc=45000, co_inc=10000, loan=150, term=360,
                              credit=CREDIT_OPTS[1], edu="Graduate", emp="No", area="Urban"),
}

# ------------------------------------------------------------------ session state
for k, v in EXAMPLES["Strong applicant"].items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("example", "Strong applicant")
st.session_state.setdefault("page", "Home")


def apply_example(name):
    st.session_state["example"] = name
    for k, v in EXAMPLES[name].items():
        st.session_state[k] = v


def go_predict():
    st.session_state["page"] = "Predict"


def head(title, sub):
    st.markdown(f'<div class="ptitle">{escape(title)}</div><div class="psub">{escape(sub)}</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown('<div class="brand">Loan Predictor</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Menu", ["Home", "Predict", "Data Insights", "About the Model"],
                        key="page", label_visibility="collapsed")
st.sidebar.markdown('<div class="side-foot">Learning project. Not a real banking decision '
                    'system.</div>', unsafe_allow_html=True)

metrics = load_json("outputs/metrics.json")
insights = load_json("outputs/insights.json")
if metrics is None or insights is None:
    st.error("Model files not found. Run `python step1_explore.py` and "
             "`python step2_train.py` first.")
    st.stop()

best = metrics["best_model"]
res = metrics["results"][best]

# ================================================================ HOME
if page == "Home":
    with st.container(key="hero"):
        st.markdown('<div class="eyebrow">MACHINE LEARNING DEMO</div>'
                    '<div class="hero-h">Will this loan be approved?</div>'
                    '<p class="hero-p">Enter applicant and loan details and get a model estimate '
                    'with a probability in seconds.</p>', unsafe_allow_html=True)
        st.button("Start predicting", key="start", type="primary", on_click=go_predict)

    st.markdown('<div class="sect">How it works</div>', unsafe_allow_html=True)
    steps = [("1", "Open Predict", "Choose Predict in the sidebar."),
             ("2", "Fill the form", "Enter details or pick an example."),
             ("3", "Press Predict", "The model scores the application."),
             ("4", "Read the result", "See the status and probability.")]
    st.markdown(grid("g4", [
        f'<div class="card step"><div class="num">{n}</div><h4>{t}</h4><p>{d}</p></div>'
        for n, t, d in steps]), unsafe_allow_html=True)
    st.markdown(NOTE, unsafe_allow_html=True)

# ================================================================ PREDICT
elif page == "Predict":
    head("Loan Approval Prediction", "Enter details or load an example, then press Predict.")

    with st.container(key="pills"):
        cols = st.columns([1.1, 1.35, 1.6, 1.75, 4])
        cols[0].markdown('<div class="pill-label">Try an example:</div>', unsafe_allow_html=True)
        for col, name in zip(cols[1:4], EXAMPLES):
            col.button(name, key=f"ex_{name}", on_click=apply_example, args=(name,),
                       type="primary" if st.session_state["example"] == name else "secondary")

    with st.container(key="form"):
        c1, c2, c3, c4 = st.columns(4)
        app_inc = c1.number_input("Applicant income", min_value=0, step=1000, key="app_inc")
        c1.markdown('<div class="hint">Most applications: 10,000 – 120,000</div>', unsafe_allow_html=True)
        co_inc = c2.number_input("Co-applicant income", min_value=0, step=1000, key="co_inc")
        c2.markdown('<div class="hint">0 if none</div>', unsafe_allow_html=True)
        loan = c3.number_input("Loan amount", min_value=1, step=5, key="loan")
        c3.markdown('<div class="hint">Most applications: 20 – 300</div>', unsafe_allow_html=True)
        term = c4.selectbox("Loan term (months)", [120, 180, 240, 300, 360], key="term")
        c4.markdown('<div class="hint">120 – 360 (10 – 30 years)</div>', unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        credit = d1.selectbox("Credit history", CREDIT_OPTS, key="credit")
        edu = d2.selectbox("Education", ["Graduate", "Not Graduate"], key="edu")
        emp = d3.selectbox("Self employed", ["No", "Yes"], key="emp")
        area = d4.selectbox("Property area", ["Urban", "Semiurban", "Rural"], key="area")
        run = st.button("Predict", key="predict_btn", type="primary")

    if run:
        row = pd.DataFrame([{
            "ApplicantIncome": app_inc, "CoapplicantIncome": co_inc, "LoanAmount": loan,
            "Loan_Amount_Term": term, "Credit_History": float(credit[0]),
            "Education": edu, "Self_Employed": emp, "Property_Area": area}])[FEATURES]
        model = load_model()
        p = float(model.predict_proba(row)[0, 1])

        if 0.40 <= p <= 0.60:
            cls, title = "mid", "Uncertain"
        elif p > 0.60:
            cls, title = "good", "Approved"
        else:
            cls, title = "bad", "Rejected"

        result = (f'<div class="res {cls}"><div class="lab">Predicted status</div>'
                  f'<div class="big">{title}</div>'
                  f'<div class="prow"><span>Probability of approval</span><b>{p:.0%}</b></div>'
                  f'<span class="track"><span class="fill" style="width:{p * 100:.0f}%"></span></span>'
                  f'<div class="rnote">Estimate based on a sample dataset. '
                  f'Between 40% and 60% the app shows “Uncertain”.</div></div>')
        st.markdown(result, unsafe_allow_html=True)

        if metrics.get("reference"):
            helped, against, what_if = explain(model, row, metrics["reference"], p)
            lines = lambda items: "".join(f"<p>{escape(i)}</p>" for i in items)
            st.markdown(grid("g3", [
                f'<div class="card explain"><h3 class="help">Raised the estimate</h3>{lines(helped)}</div>',
                f'<div class="card explain"><h3 class="against">Lowered the estimate</h3>{lines(against)}</div>',
                f'<div class="card explain"><h3>What if?</h3><p>{escape(what_if)}</p></div>',
            ]), unsafe_allow_html=True)
            st.caption("Each detail is compared with a typical applicant in the dataset. This shows "
                       "how the model reacts, not a bank's reasons.")
        st.markdown(NOTE, unsafe_allow_html=True)

# ================================================================ DATA INSIGHTS
elif page == "Data Insights":
    ins = insights
    pct = lambda d: [(k, v * 100, f"{v:.0%}") for k, v in d.items()]
    spread = lambda d: max(d.values()) - min(d.values())
    credit, income = ins["credit"], ins["income"]

    head("Data Insights", f"Approval rates in the {ins['n_rows']:,} cleaned applications. "
                          f"Overall: {ins['approval_rate']:.0%} approved.")
    st.markdown(grid("g3", [
        stat(f"{ins['n_rows']:,}", f"applications after removing {ins['n_duplicates']:,} duplicates"),
        stat(f"{ins['approval_rate']:.0%}", "approved"),
        stat(f"{credit['0']:.0%} vs {credit['1']:.0%}", "approval without vs with good credit history"),
    ]), unsafe_allow_html=True)

    inc_vals = list(income.values())
    rising = all(a <= b for a, b in zip(inc_vals, inc_vals[1:]))
    strongest = spread(credit) >= max(spread(income), spread(ins["education"]), spread(ins["area"]))
    same = lambda d: "Almost no difference." if spread(d) < 0.05 else "A noticeable difference."

    st.markdown(grid("g2", [
        card("By credit history",
             bar_rows([("No / bad (0)", credit["0"] * 100, f"{credit['0']:.0%}"),
                       ("Good (1)", credit["1"] * 100, f"{credit['1']:.0%}")]),
             "The strongest signal in the data." if strongest else ""),
        card("By income band", bar_rows(pct(income)),
             "Approval rises steadily with income." if rising else "Approval varies with income."),
        card("By education", bar_rows(pct(ins["education"])), same(ins["education"])),
        card("By property area", bar_rows(pct(ins["area"])), same(ins["area"])),
    ]), unsafe_allow_html=True)

    st.markdown(card(
        "Data cleaning summary",
        f'<p>{ins["n_duplicates"]:,} duplicate rows removed. Messy labels merged (for example '
        f'“Semi-Urban” and “urban”). {ins["n_invalid_loan"]:,} loan amounts of 0 or less treated '
        f'as missing. Missing values filled inside the model pipeline.</p>'), unsafe_allow_html=True)

# ================================================================ ABOUT THE MODEL
else:
    cm = res["confusion_matrix"]
    shorten = {"Logistic Regression": "Logistic Reg."}
    ranked = sorted(metrics["results"].items(), key=lambda kv: -kv[1]["test_roc_auc"])
    best_cv = max(metrics["results"], key=lambda n: metrics["results"][n]["cv_roc_auc"])

    st.markdown('<div class="ptitle">About the Model</div>', unsafe_allow_html=True)

    st.markdown(grid("g4", [
        stat(f"{res['test_roc_auc']:.2f}", "ROC-AUC"),
        stat(f"{res['recall_approved']:.0%}", "Approved recall"),
        stat(f"{res['recall_rejected']:.0%}", "Rejected recall"),
        stat(f"{res['precision_rejected']:.0%}", "Rejected precision"),
    ]), unsafe_allow_html=True)

    comp = bar_rows([(shorten.get(n, n), r["test_roc_auc"] * 100, f"{r['test_roc_auc'] * 100:.0f}")
                     for n, r in ranked])
    pr = bar_rows([
        ("Rejected prec.", res["precision_rejected"] * 100, f"{res['precision_rejected']:.0%}"),
        ("Rejected recall", res["recall_rejected"] * 100, f"{res['recall_rejected']:.0%}"),
        ("Rejected F1", res["f1_rejected"] * 100, f"{res['f1_rejected']:.0%}"),
        ("Approved prec.", res["precision_approved"] * 100, f"{res['precision_approved']:.0%}"),
        ("Approved recall", res["recall_approved"] * 100, f"{res['recall_approved']:.0%}"),
        ("Approved F1", res["f1_approved"] * 100, f"{res['f1_approved']:.0%}"),
    ], color="gold")
    imp = metrics["feature_importance"]
    imp_card = (card("Feature importance",
                     bar_rows([(k, v, f"{v:.0f}") for k, v in list(imp.items())[:5]]),
                     "Shows what the model used, not whether it is fair.")
                if imp else card("Feature importance", "<p>Not available for this model.</p>"))

    st.markdown(grid("g2", [
        card("Model comparison (ROC-AUC, test)", comp,
             f"{best_cv} had the best cross-validated score "
             f"({metrics['results'][best_cv]['cv_roc_auc']:.2f})."),
        card("Confusion matrix", charts.confusion(cm)),
        card("Precision / Recall / F1", pr),
        imp_card,
    ]), unsafe_allow_html=True)