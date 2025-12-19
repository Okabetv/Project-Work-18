from __future__ import annotations
import re
from typing import Optional, Tuple

# Keyword “forti” (override)
HIGH_RULES = [
    r"\bbloccante\b", r"\burgente\b", r"\bcritico\b", r"\bcrash\b",
    r"\berrore 500\b", r"\bnon disponibile\b", r"\bapi non risponde\b"
]
MEDIUM_RULES = [
    r"\btimeout\b", r"\blent[oaie]\b", r"\bin ritardo\b", r"\bmancante\b", r"\bnon corretta\b"
]

CONF_LOW = 0.55  # sotto questa soglia: bassa confidenza


def rule_priority(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(re.search(p, t) for p in HIGH_RULES):
        return "alta"
    if any(re.search(p, t) for p in MEDIUM_RULES):
        return "media"
    return None


def predict_priority_hybrid(priority_model, text: str) -> Tuple[str, Optional[float], str]:
    """
    Ritorna: (priorità_finale, confidenza_ml, motivo)
    motivo: 'rule_high', 'rule_medium', 'ml', 'ml_low_conf'
    """
    t = (text or "").strip()

    # 1) Regole
    rp = rule_priority(t)
    if rp == "alta":
        return "alta", None, "rule_high"
    if rp == "media":
        return "media", None, "rule_medium"

    # 2) ML
    pred = priority_model.predict([t])[0]
    proba = None
    if hasattr(priority_model, "predict_proba"):
        probs = priority_model.predict_proba([t])[0]
        proba = float(max(probs))

    # 3) Se confidenza bassa: comportamento conservativo
    if proba is not None and proba < CONF_LOW:
        # conservativo: non alzare troppo. Se ML dice alta ma è insicuro → media
        if pred == "alta":
            return "media", proba, "ml_low_conf"
        return pred, proba, "ml_low_conf"

    return pred, proba, "ml"
