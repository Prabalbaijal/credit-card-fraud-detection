import os
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_data, split_features_target, scale_features

def load_models(models_dir="models"):
    models = {}
    for name in ["logistic_regression.pkl", "random_forest.pkl", "xgboost.pkl"]:
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            models[name.replace(".pkl", "")] = joblib.load(path)
    return models

def evaluate_single(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob)
    }
    return metrics, confusion_matrix(y_test, y_pred)

def compare_models(models, X_test, y_test, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for name, m in models.items():
        metrics, cm = evaluate_single(m, X_test, y_test)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(os.path.join(results_dir, f"{name}_cm.png"))
        plt.close()
        rows.append({"model": name, **metrics})
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    return df

if __name__ == "__main__":
    CSV_PATH = "data/creditcard.csv"
    df = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = split_features_target(df)
    _, X_test_scaled, _ = scale_features(X_train, X_test, scaler_path="models/scaler.gz")
    models = load_models("models")
    results = compare_models(models, X_test_scaled, y_test)
    print(results)
