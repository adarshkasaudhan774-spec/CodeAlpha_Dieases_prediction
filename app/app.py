import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.data_loader import LOADERS
from src.predict import predict_patient

st.set_page_config(page_title="Disease Prediction", page_icon="🩺")
st.title("🩺 Disease Prediction")
st.caption("Educational project. Not a medical diagnosis tool.")

dataset = st.selectbox("Disease", ["heart", "diabetes", "breast"],
                       format_func=lambda d: {"heart": "Heart disease", "diabetes": "Diabetes",
                                              "breast": "Breast cancer"}[d])


@st.cache_data
def get_stats(name):
    X, _ = LOADERS[name]()
    return X.describe().T[["min", "50%", "max"]]


stats = get_stats(dataset)
patient = {}
cols = st.columns(2)
for i, (feat, row) in enumerate(stats.iterrows()):
    patient[feat] = cols[i % 2].number_input(
        feat, min_value=float(row["min"]), max_value=float(row["max"]), value=float(row["50%"]))

if st.button("Predict"):
    pred, prob = predict_patient(dataset, patient)
    if pred:
        st.error(f"Higher risk detected (probability {prob:.1%})")
    else:
        st.success(f"Lower risk (probability of disease {prob:.1%})")