import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "yc_startup_validator_app" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv("data/cleaned/yc_companies_clean.csv")
df = df.dropna(subset=["description", "company_size_bucket"])

# Binary success label
df["success"] = df["company_size_bucket"].isin(["scaling", "large"]).astype(int)

# Vectorize
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.8,
    min_df=10,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(df["description"].str.lower())

# Cluster
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

df["cluster"] = clusters

# Predict success
clf = LogisticRegression(max_iter=1000)
clf.fit(X, df["success"])

# Save artifacts
pickle.dump(vectorizer, open(MODEL_DIR / "vectorizer.pkl", "wb"))
pickle.dump(kmeans, open(MODEL_DIR / "kmeans.pkl", "wb"))
pickle.dump(clf, open(MODEL_DIR / "success_model.pkl", "wb"))

print("✅ Model artifacts saved to app/model/")
