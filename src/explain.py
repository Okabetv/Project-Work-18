from __future__ import annotations

import numpy as np


def _top_from_vector(vec, feature_names, scores, k: int):
    """Ritorna top-k feature tra quelle presenti nel testo (vec>0), ordinate per score desc."""
    present = vec > 0
    if not np.any(present):
        return []

    idx = np.argsort(scores)[::-1]
    idx = [i for i in idx if present[i]][:k]
    return list(zip(feature_names[idx].tolist(), scores[idx].tolist()))


def top_terms(pipe, text: str, k: int = 5):
    """
    Estrae le top-k parole/frasi più influenti per la classe predetta.
    Supporta:
      - LogisticRegression (coef_)
      - MultinomialNB (feature_log_prob_)
    Restituisce: (pred_class, [(term, score), ...])
    """
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    X = tfidf.transform([text])
    pred = clf.predict(X)[0]

    feature_names = np.array(tfidf.get_feature_names_out())
    vec = X.toarray().ravel()

    classes = list(getattr(clf, "classes_", []))
    if pred in classes:
        cidx = classes.index(pred)
    else:
        cidx = 0

    # Caso 1: Logistic Regression
    if hasattr(clf, "coef_"):
        coefs = clf.coef_[cidx] if clf.coef_.ndim == 2 else clf.coef_
        scores = vec * coefs
        return pred, _top_from_vector(vec, feature_names, scores, k)

    # Caso 2: Multinomial Naive Bayes
    # Usiamo: score ~ tfidf(term) * log P(term | class_predetta)
    if hasattr(clf, "feature_log_prob_"):
        logp = clf.feature_log_prob_[cidx]  # shape: (n_features,)
        scores = vec * logp
        return pred, _top_from_vector(vec, feature_names, scores, k)

    # Fallback
    return pred, [("(spiegabilità non disponibile per questo modello)", 0.0)]
