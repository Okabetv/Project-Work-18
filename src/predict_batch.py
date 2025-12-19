import joblib
import pandas as pd

from src.priority_hybrid import predict_priority_hybrid


def main(in_csv="data/tickets.csv", out_csv="data/predictions.csv"):
    df = pd.read_csv(in_csv)
    X = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)

    cat_model = joblib.load("models/category_model.joblib")
    pri_model = joblib.load("models/priority_model.joblib")

    out = df.copy()
    out["pred_category"] = cat_model.predict(X)

    if hasattr(cat_model, "predict_proba"):
        out["prob_category"] = cat_model.predict_proba(X).max(axis=1)

    preds = []
    probs = []
    reasons = []
    for txt in X.tolist():
        p, pr, reason = predict_priority_hybrid(pri_model, txt)
        preds.append(p)
        probs.append(pr)
        reasons.append(reason)

    out["pred_priority"] = preds
    out["prob_priority_ml"] = probs
    out["priority_reason"] = reasons

    out.to_csv(out_csv, index=False)
    print(f"Creato: {out_csv} ({len(out)} righe)")


if __name__ == "__main__":
    main()
