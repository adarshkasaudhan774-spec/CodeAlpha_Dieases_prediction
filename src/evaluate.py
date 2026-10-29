"""Metrics and plots."""
import matplotlib
matplotlib.use("Agg")   # save figures without opening windows
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.data_loader import FIGURES_DIR


def compute_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),      # sensitivity: most important in medical tasks
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def plot_confusion_matrix(model, X_test, y_test, title, filename):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def plot_roc_curves(fitted_models, X_test, y_test, dataset):
    """fitted_models: {model_name: fitted pipeline}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(f"{dataset}: ROC curves")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{dataset}_roc.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, dataset, top_n=10):
    """Works for RandomForest / XGBoost pipelines."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    names = model.named_steps["prep"].get_feature_names_out()
    imp = clf.feature_importances_
    order = imp.argsort()[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order][::-1], imp[order][::-1], color="#4C72B0")
    ax.set_title(f"{dataset}: top {top_n} feature importances")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{dataset}_feature_importance.png", dpi=150)
    plt.close(fig)