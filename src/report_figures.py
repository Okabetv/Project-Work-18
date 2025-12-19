from __future__ import annotations

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def save_bar_counts(series: pd.Series, title: str, out_path: str):
    counts = series.value_counts().sort_index()
    ax = counts.plot(kind="bar")
    ax.set_title(title)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Conteggio")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_f1_per_class(y_true, y_pred, title: str, out_path: str):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # Filtra solo classi (evita accuracy/macro avg/weighted avg)
    rows = {k: v for k, v in rep.items() if isinstance(v, dict) and "f1-score" in v}
    f1 = pd.Series({k: rows[k]["f1-score"] for k in rows}).sort_index()

    ax = f1.plot(kind="bar")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Classe")
    ax.set_ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv("data/tickets.csv")

    # 1) Distribuzione classi
    save_bar_counts(df["category"], "Distribuzione classi - Categoria", "reports/class_distribution_category.png")
    save_bar_counts(df["priority"], "Distribuzione classi - Priorità", "reports/class_distribution_priority.png")

    # 2) F1 per classe su split 80/20 (usiamo i modelli salvati)
    X = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)

    # Categoria
    y_cat = df["category"].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
    cat_model = joblib.load("models/category_model.joblib")
    ypred_cat = cat_model.predict(Xte)
    save_f1_per_class(yte, ypred_cat, "F1 per classe - Categoria (test 20%)", "reports/f1_per_class_category.png")

    # Priorità
    y_pri = df["priority"].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y_pri, test_size=0.2, random_state=42, stratify=y_pri)
    pri_model = joblib.load("models/priority_model.joblib")
    ypred_pri = pri_model.predict(Xte)
    save_f1_per_class(yte, ypred_pri, "F1 per classe - Priorità (test 20%)", "reports/f1_per_class_priority.png")

    print("Creati grafici in reports/: distribuzioni + F1 per classe")

if __name__ == "__main__":
    main()
