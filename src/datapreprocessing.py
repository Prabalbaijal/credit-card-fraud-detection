# src/data_preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=None):
    """Load CSV (returns DataFrame)."""
    if path is None:
        path = os.path.join(DATA_DIR, "creditcard.csv")
    df = pd.read_csv(path)
    return df

def basic_eda(df, n=5):
    """Return basic EDA info."""
    return {
        "shape": df.shape,
        "head": df.head(n),
        "tail": df.tail(n),
        "dtypes": df.dtypes,
        "missing": df.isnull().sum(),
        "class_counts": df['Class'].value_counts()
    }

def scale_features(X_train, X_test, scaler_path=None):
    """Standardize features (fit on train, transform on test)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled, scaler

def split_features_target(df, target_col='Class', test_size=0.2, random_state=42, stratify=True):
    """Split dataset into train/test with optional stratification."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train, random_state=42):
    """Balance classes using SMOTE."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
