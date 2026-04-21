"""
Component 2 — MLflow experiment tracking
Mirrors scripts/train_model.py but logs every run to MLflow.
Run from project root: python3.11 phase2/lifecycle/train_tracked.py
"""

import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "yc_startup_validator_app" / "model"
DATA_PATH = BASE_DIR / "data" / "cleaned" / "yc_companies_clean.csv"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("yc-startup-validator")

df = pd.read_csv(DATA_PATH).dropna(subset=["description", "company_size_bucket"])
df["success"] = df["company_size_bucket"].isin(["scaling", "large"]).astype(int)

params = {
    "max_df": 0.8,
    "min_df": 10,
    "ngram_range": "(1,2)",
    "max_iter": 1000,
}

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_param("train_size", len(df))
    mlflow.log_param("positive_rate", round(df["success"].mean(), 4))

    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.8, min_df=10, ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["description"].str.lower())
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "n_features": X.shape[1],
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(clf, "model", registered_model_name="yc-success-classifier")
    mlflow.sklearn.log_model(vectorizer, "vectorizer")

    # Overwrite pkl artifacts so the API stays in sync
    pickle.dump(vectorizer, open(MODEL_DIR / "vectorizer.pkl", "wb"))
    pickle.dump(clf, open(MODEL_DIR / "success_model.pkl", "wb"))

    print(f"accuracy={metrics['accuracy']} f1={metrics['f1']} roc_auc={metrics['roc_auc']}")
    print("Run logged to MLflow. Artifacts saved to model/")
