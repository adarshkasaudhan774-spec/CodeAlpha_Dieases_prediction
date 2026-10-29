"""Predict for a single patient."""
import joblib
import pandas as pd

from src.data_loader import MODELS_DIR


def load_model(dataset, which="best"):
    return joblib.load(MODELS_DIR / f"{dataset}_{which}.joblib")


def predict_patient(dataset, patient: dict, which="best"):
    """patient: {feature_name: value}. Missing features are imputed."""
    model = load_model(dataset, which)
    df = pd.DataFrame([patient])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0, 1])
    return pred, prob


if __name__ == "__main__":
    example = {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
               "restecg": 0, "thalach": 120, "exang": 1, "oldpeak": 2.0,
               "slope": 2, "ca": 1, "thal": 7}
    pred, prob = predict_patient("heart", example)
    print(f"Prediction: {'Disease' if pred else 'No disease'}  (probability {prob:.2%})")