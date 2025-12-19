from __future__ import annotations

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.features import basic_clean


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(preprocessor=basic_clean, ngram_range=(1, 2), min_df=2)


def eval_model(name: str, pipe: Pipeline, X_train, X_test, y_train, y_test, label_col: str) -> dict:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    print(f"\n== {label_col.upper()} | {name} ==")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 macro: {f1m:.3f}")
    print(classification_report(y_test, y_pred))

    os.makedirs("reports", exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=25)
    plt.title(f"Confusion Matrix - {label_col} - {name}")
    plt.savefig(f"reports/confusion_{label_col}_{name}.png", bbox_inches="tight")
    plt.close()

    return {"name": name, "pipe": pipe, "accuracy": acc, "f1_macro": f1m}


def train_category(df: pd.DataFrame) -> dict:
    X = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)
    y = df["category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = build_vectorizer()

    # Modello 1: Logistic Regression
    pipe_lr = Pipeline([("tfidf", vec), ("clf", LogisticRegression(max_iter=2000))])
    res_lr = eval_model("LogReg", pipe_lr, X_train, X_test, y_train, y_test, "category")

    # Modello 2: Multinomial Naive Bayes
    # NB richiede feature non-negative => TF-IDF ok
    pipe_nb = Pipeline([("tfidf", build_vectorizer()), ("clf", MultinomialNB())])
    res_nb = eval_model("NaiveBayes", pipe_nb, X_train, X_test, y_train, y_test, "category")

    best = max([res_lr, res_nb], key=lambda r: r["f1_macro"])
    print(f"\n>>> Miglior modello CATEGORY: {best['name']} (F1 macro={best['f1_macro']:.3f})")
    return best


def train_priority(df: pd.DataFrame) -> dict:
    X = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)
    y = df["priority"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    res = eval_model("LogReg", pipe, X_train, X_test, y_train, y_test, "priority")
    return res


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv("data/tickets.csv")

    best_cat = train_category(df)
    pri_res = train_priority(df)

    # Salva SOLO il best per category
    joblib.dump(best_cat["pipe"], "models/category_model.joblib")
    joblib.dump(pri_res["pipe"], "models/priority_model.joblib")

    # Riassunto metriche
    with open("reports/metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write("PW18 - Sintesi metriche\n\n")
        f.write(f"Categoria - Best: {best_cat['name']} | Acc: {best_cat['accuracy']:.3f} | F1 macro: {best_cat['f1_macro']:.3f}\n")
        f.write(f"Priorità - LogReg | Acc: {pri_res['accuracy']:.3f} | F1 macro: {pri_res['f1_macro']:.3f}\n")

    print("\nSalvati modelli in /models e grafici in /reports")
    print("Nota: confusion matrix anche per entrambi i modelli categoria (NB e LogReg).")


if __name__ == "__main__":
    main()
