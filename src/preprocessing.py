"""Preprocessing pipeline: imputation, scaling, encoding, train/test split."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import PROCESSED_DIR

# Categorical columns per dataset (everything else is treated as numeric)
CATEGORICAL = {
    "heart": ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    "breast": [],
    "diabetes": [],
}


def build_preprocessor(X, dataset):
    cat_cols = [c for c in CATEGORICAL[dataset] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers = [("num", num_pipe, num_cols)]

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers)


def split_data(X, y, test_size=0.2, seed=42):
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def save_splits(name, X_train, X_test, y_train, y_test):
    """Save the raw (unprocessed) splits to Data/Processed for reference."""
    pd.concat([X_train, y_train.rename("target")], axis=1).to_csv(PROCESSED_DIR / f"{name}_train.csv", index=False)
    pd.concat([X_test, y_test.rename("target")], axis=1).to_csv(PROCESSED_DIR / f"{name}_test.csv", index=False)