"""Model definitions and hyperparameter grids (prefix 'clf__' = pipeline step name)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_models(seed=42):
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=seed),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"], "clf__gamma": ["scale", "auto"]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=seed),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 3]},
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=seed),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8, 1.0]},
        ),
    }