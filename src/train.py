"""Train and tune all models on all datasets. Run from project root:  python -m src.train"""
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_loader import MODELS_DIR, REPORTS_DIR, load_all
from src.evaluate import (compute_metrics, plot_confusion_matrix,
                          plot_feature_importance, plot_roc_curves)
from src.models import get_models
from src.preprocessing import build_preprocessor, save_splits, split_data

SEED = 42


def train_dataset(name, X, y):
    X_train, X_test, y_train, y_test = split_data(X, y, seed=SEED)
    save_splits(name, X_train, X_test, y_train, y_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}

    for model_name, (clf, grid) in get_models(SEED).items():
        print(f"  training {model_name} ...")
        pipe = Pipeline([("prep", build_preprocessor(X_train, name)), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        fitted[model_name] = best
        metrics = compute_metrics(best, X_test, y_test)
        rows.append({"dataset": name, "model": model_name,
                     "cv_auc": gs.best_score_, **metrics, "best_params": gs.best_params_})

        joblib.dump(best, MODELS_DIR / f"{name}_{model_name}.joblib")
        plot_confusion_matrix(best, X_test, y_test, f"{name} - {model_name}",
                              f"{name}_{model_name}_cm.png")
        if model_name in ("RandomForest", "XGBoost"):
            plot_feature_importance(best, X.columns, f"{name}_{model_name}")

    plot_roc_curves(fitted, X_test, y_test, name)

    # Choose the best model by cross-validated AUC (not by test score)
    results = pd.DataFrame(rows)
    best_name = results.loc[results["cv_auc"].idxmax(), "model"]
    joblib.dump(fitted[best_name], MODELS_DIR / f"{name}_best.joblib")
    print(f"  best for {name}: {best_name}")
    return results


if __name__ == "__main__":
    all_results = []
    for name, (X, y) in load_all().items():
        print(f"\n=== {name.upper()} ===")
        all_results.append(train_dataset(name, X, y))

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(REPORTS_DIR / "results.csv", index=False)
    show = final.drop(columns="best_params").round(3)
    print("\n", show.to_string(index=False))