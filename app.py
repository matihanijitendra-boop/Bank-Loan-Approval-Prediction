import streamlit as st
import pandas as pd
from pathlib import Path
from train_model import load_or_train_model, FEATURES

st.set_page_config(page_title="Bank Loan Approval Prediction", page_icon="🏦", layout="wide")
st.title("🏦 Bank Loan Approval Prediction")
st.caption("A machine-learning classification demo using a Random Forest model.")

@st.cache_resource
def get_model():
    return load_or_train_model()

try:
    model, metrics = get_model()
except Exception as exc:
    st.error(f"Could not load/train the model: {exc}")
    st.stop()

with st.sidebar:
    st.header("About this project")
    st.write("The model predicts whether a loan application is likely to be approved (Y) or rejected (N).")
    st.write("Model: Random Forest")
    st.write("Evaluation: train/test split and confusion matrix")
    st.warning("Educational demo only. Do not use this prediction to make real lending decisions.")

st.subheader("Enter applicant details")
c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
with c2:
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=150, step=10)
with c3:
    loan_term = st.selectbox("Loan Amount Term (months)", [120, 180, 240, 300, 360, 480], index=4)
    credit_history = st.selectbox("Credit History", ["Good (1)", "Poor (0)", "Unknown"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Loan Status", type="primary", use_container_width=True):
    credit_value = {"Good (1)": 1.0, "Poor (0)": 0.0, "Unknown": None}[credit_history]
    applicant = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_value, "Property_Area": property_area
    }])[FEATURES]
    prediction = model.predict(applicant)[0]
    probability = model.predict_proba(applicant)[0]
    classes = list(model.named_steps["classifier"].classes_)
    approval_probability = float(probability[classes.index("Y")]) * 100
    if prediction == "Y":
        st.success(f"Prediction: Likely Approved — estimated model probability {approval_probability:.1f}%")
    else:
        st.error(f"Prediction: Likely Not Approved — estimated approval probability {approval_probability:.1f}%")
    st.progress(int(round(approval_probability)))

st.divider()
st.subheader("Model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
m2.metric("Test samples", str(metrics["test_samples"]))
m3.metric("Training samples", str(metrics["train_samples"]))
st.write("Confusion matrix (rows = actual class, columns = predicted class):")
st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["Actual N", "Actual Y"],
                          columns=["Predicted N", "Predicted Y"]), use_container_width=True)
st.caption("The included CSV is a synthetic demo dataset. Replace it with the official Kaggle dataset for assignment results.")
