"""
Component 5 — Automated retraining pipeline
Fetches latest YC data, retrains, compares against best prior run.
Promotes (overwrites pkl artifacts) only if new model beats current f1.
Run from project root: python3.11 phase2/lifecycle/retrain.py
"""

import json
import pickle
import urllib.request
import pandas as pd
import mlflow
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "yc_startup_validator_app" / "model"
YC_API = "https://yc-oss.github.io/api/companies/all.json"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("yc-startup-validator")


def fetch_data() -> pd.DataFrame:
    print("Fetching latest YC data...")
    with urllib.request.urlopen(YC_API) as r:
        raw = json.loads(r.read())
    df = pd.DataFrame(raw)
    df = df.rename(columns={"one_liner": "description", "team_size": "company_size_num"})
    df = df.dropna(subset=["description"])
    df["success"] = df["company_size_num"].apply(lambda x: 1 if x and x >= 51 else 0)
    print(f"Fetched {len(df)} companies. Positive rate: {df['success'].mean():.2%}")
    return df


def get_best_f1() -> float:
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("yc-startup-validator").experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1,
    )
    if not runs:
        return 0.0
    best = runs[0].data.metrics.get("f1", 0.0)
    print(f"Current best f1 in MLflow: {best:.4f}")
    return best


def train(df: pd.DataFrame):
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
    f1 = round(f1_score(y_test, clf.predict(X_test)), 4)
    roc = round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]), 4)
    return clf, vectorizer, f1, roc


def promote(clf, vectorizer):
    pickle.dump(vectorizer, open(MODEL_DIR / "vectorizer.pkl", "wb"))
    pickle.dump(clf, open(MODEL_DIR / "success_model.pkl", "wb"))
    print("Promoted — pkl artifacts updated.")


if __name__ == "__main__":
    df = fetch_data()
    best_f1 = get_best_f1()

    with mlflow.start_run():
        mlflow.log_param("train_size", len(df))
        mlflow.log_param("source", "yc_api_live")

        clf, vectorizer, f1, roc = train(df)
        mlflow.log_metrics({"f1": f1, "roc_auc": roc})
        mlflow.sklearn.log_model(clf, "model")

        print(f"New model — f1={f1} roc_auc={roc}")

        if f1 > best_f1:
            print(f"Improvement: {best_f1:.4f} → {f1:.4f}. Promoting.")
            promote(clf, vectorizer)
        else:
            print(f"No improvement over {best_f1:.4f}. Keeping current model.")
