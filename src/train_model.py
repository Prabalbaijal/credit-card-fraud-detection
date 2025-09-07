import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.data_preprocessing import load_data, split_features_target, scale_features, apply_smote

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_models(csv_path, scaler_path=None):
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test = split_features_target(df, target_col='Class', test_size=0.2)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scaler_path=scaler_path)
    if scaler_path:
        joblib.dump(scaler, scaler_path)

    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
    lr.fit(X_train_res, y_train_res)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train_res, y_train_res)
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))

    # XGBoost
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = max(1.0, n_neg / max(1, n_pos))

    xgb_clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    )
    xgb_clf.fit(X_train_scaled, y_train)
    joblib.dump(xgb_clf, os.path.join(MODEL_DIR, "xgboost.pkl"))

    return {
        "logistic": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
        "random_forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
        "xgboost": os.path.join(MODEL_DIR, "xgboost.pkl"),
    }

if __name__ == "__main__":
    CSV_PATH = "data/creditcard.csv"
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.gz")
    train_and_save_models(CSV_PATH, scaler_path=SCALER_PATH)
