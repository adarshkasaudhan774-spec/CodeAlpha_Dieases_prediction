"""Load the three disease datasets and define project paths."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
MODELS_DIR = PROJECT_ROOT / "Models"
REPORTS_DIR = PROJECT_ROOT / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _fetch_uci(uci_id, name):
    """Download a UCI dataset once, cache it as CSV in Data/Raw, reuse later."""
    cache = RAW_DIR / f"{name}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=uci_id)
    df = pd.concat([d.data.features, d.data.targets], axis=1)
    df.to_csv(cache, index=False)
    return df


def load_heart():
    """UCI Heart Disease (id=45). Target 'num' 0-4 -> binary (0 = no disease, 1 = disease)."""
    df = _fetch_uci(45, "heart")
    return df.drop(columns="num"), (df["num"] > 0).astype(int)


def load_breast():
    """UCI Breast Cancer Wisconsin Diagnostic (id=17). M = 1, B = 0."""
    df = _fetch_uci(17, "breast")
    return df.drop(columns="Diagnosis"), (df["Diagnosis"] == "M").astype(int)


def load_diabetes(path=None):
    """Pima Indians Diabetes: expects Data/Raw/diabetes.csv."""
    path = Path(path) if path else RAW_DIR / "diabetes.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put diabetes.csv in Data/Raw/.")

    df = pd.read_csv(path)
    if "Outcome" not in df.columns:   # file without a header row
        df = pd.read_csv(path, header=None, names=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])

    # Zeros in these columns are impossible values, so treat them as missing
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

    return df.drop(columns="Outcome"), df["Outcome"]


LOADERS = {
    "heart": load_heart,
    "breast": load_breast,
    "diabetes": load_diabetes,
}


def load_all():
    """Return {name: (X, y)} for all three datasets."""
    return {name: fn() for name, fn in LOADERS.items()}


if __name__ == "__main__":
    for name, (X, y) in load_all().items():
        print(f"{name}: X={X.shape}, positives={y.mean():.2%}, missing={int(X.isna().sum().sum())}")