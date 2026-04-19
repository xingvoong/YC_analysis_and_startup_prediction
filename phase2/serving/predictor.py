"""
Core prediction logic decoupled from the API layer.
Can be tested independently of FastAPI.
"""

import numpy as np
from typing import Tuple, List
from schemas import SignalWord


def interpret_score(score: int) -> Tuple[str, str]:
    if score >= 75:
        return "Very strong YC-style signal", "strong"
    elif score >= 55:
        return "Promising, but highly competitive", "promising"
    elif score >= 35:
        return "Weak signal — niche or execution-heavy", "weak"
    else:
        return "Low historical success similarity", "low"


def explain_prediction(vectorizer, model, X, top_k: int = 8) -> Tuple[List[SignalWord], List[SignalWord]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    x_dense = X.toarray()[0]
    contributions = x_dense * coefs

    top_positive_idx = contributions.argsort()[-top_k:][::-1]
    top_negative_idx = contributions.argsort()[:top_k]

    positive_signals = [
        SignalWord(word=feature_names[i], contribution=round(float(contributions[i]), 4))
        for i in top_positive_idx
        if contributions[i] > 0
    ]
    negative_signals = [
        SignalWord(word=feature_names[i], contribution=round(float(contributions[i]), 4))
        for i in top_negative_idx
        if contributions[i] < 0
    ]

    return positive_signals, negative_signals


def run_prediction(vectorizer, model, description: str, top_k: int, version: str):
    X = vectorizer.transform([description.lower()])
    prob = model.predict_proba(X)[0][1]
    score = int(prob * 100)
    label, level = interpret_score(score)
    positive_signals, negative_signals = explain_prediction(vectorizer, model, X, top_k)

    return {
        "score": score,
        "probability": round(float(prob), 4),
        "label": label,
        "level": level,
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "model_version": version,
    }
