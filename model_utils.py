"""Training + loading helpers. The model is (re)built automatically when needed."""
import os
import pickle

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.naive_bayes import MultinomialNB

from preprocess import transform_text

BASE = os.path.dirname(os.path.abspath(__file__))
FILES = {n: os.path.join(BASE, n) for n in ("vectorizer.pkl", "model.pkl", "meta.pkl")}


def train_and_save():
    print("Training model (one time, takes ~20 seconds)...")
    df = pd.read_csv(os.path.join(BASE, "spam.csv"), encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "target", "v2": "text"})
    df["target"] = (df["target"] == "spam").astype(int)
    df = df.drop_duplicates(keep="first")
    df["transformed_text"] = df["text"].apply(transform_text)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["transformed_text"]).toarray()
    y = df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=2)

    model = MultinomialNB().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    meta = {
        "model": "MultinomialNB",
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "sklearn": sklearn.__version__,
        "messages": int(len(df)),
    }
    for name, obj in (("vectorizer.pkl", tfidf), ("model.pkl", model), ("meta.pkl", meta)):
        with open(FILES[name], "wb") as f:
            pickle.dump(obj, f)
    print(f"Done. accuracy={meta['accuracy']:.4f} precision={meta['precision']:.4f}")
    return tfidf, model, meta


def load_or_train():
    """Load saved model; retrain if missing, broken, or built with another scikit-learn version."""
    try:
        with open(FILES["meta.pkl"], "rb") as f:
            meta = pickle.load(f)
        if meta.get("sklearn") != sklearn.__version__:
            raise RuntimeError("scikit-learn version changed")
        with open(FILES["vectorizer.pkl"], "rb") as f:
            tfidf = pickle.load(f)
        with open(FILES["model.pkl"], "rb") as f:
            model = pickle.load(f)
        tfidf.transform(["test"])  # raises if the vectorizer is not fitted/usable
        return tfidf, model, meta
    except Exception as e:
        print("Saved model not usable (%s) -> retraining." % e)
        return train_and_save()
