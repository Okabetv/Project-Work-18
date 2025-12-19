import os
import sys
from datetime import datetime
import glob
import csv

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.priority_hybrid import predict_priority_hybrid, CONF_LOW
from src.explain import top_terms


st.set_page_config(page_title="STT – Smart Ticket Triage", layout="centered")

CONFIDENCE_WARN = 0.55
LOG_PATH = "data/prediction_log.csv"


@st.cache_resource
def load_models():
    cat = joblib.load("models/category_model.joblib")
    pri = joblib.load("models/priority_model.joblib")
    return cat, pri


def predict_with_proba(pipe, text: str):
    pred = pipe.predict([text])[0]
    proba = None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba([text])[0]
        proba = float(np.max(probs))
    return pred, proba


def append_log(row: dict):
    os.makedirs("data", exist_ok=True)
    df_row = pd.DataFrame([row])

    write_header = not os.path.exists(LOG_PATH)
    df_row.to_csv(
        LOG_PATH,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def load_metrics_text():
    for path in ["reports/metrics.txt", "reports/metrics_summary.txt"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return None


# ---------------- UI ----------------
st.title("STT – Smart Ticket Triage")
st.caption(
    "Inserisci un ticket breve: il sistema propone **categoria** e **priorità**. "
    "Dati sintetici, nessun dato personale."
)

cat_model, pri_model = load_models()
tab1, tab2, tab3, tab4 = st.tabs(["🧾 Classifica", "📦 Batch CSV", "📊 Metriche", "ℹ️ Info"])


# ---------------- TAB 1: CLASSIFICA ----------------
with tab1:
    st.subheader("Classificazione singolo ticket")

    if "title_demo" not in st.session_state:
        st.session_state["title_demo"] = "Errore 500 su login"
    if "body_demo" not in st.session_state:
        st.session_state["body_demo"] = "Da stamattina vedo errore HTTP 500 quando provo ad accedere. È bloccante."

    colA, colB = st.columns([3, 1])
    with colA:
        title = st.text_input("Oggetto", value=st.session_state["title_demo"])
    with colB:
        st.write("")
        st.write("")
        if st.button("🎲 Esempio", help="Carica un ticket casuale dal dataset"):
            try:
                df = pd.read_csv("data/tickets.csv")
                r = df.sample(1).iloc[0]
                st.session_state["title_demo"] = str(r["title"])
                st.session_state["body_demo"] = str(r["body"])
                st.rerun()
            except Exception:
                st.warning("Non riesco a leggere data/tickets.csv. Genera prima il dataset.")

    body = st.text_area(
        "Descrizione",
        value=st.session_state["body_demo"],
        height=130
    )

    text = (title + " " + body).strip()

    if st.button("Classifica", type="primary"):
        pred_cat, p_cat = predict_with_proba(cat_model, text)

        pred_pri, p_pri, pri_reason = predict_priority_hybrid(pri_model, text)

        st.markdown("### Risultato")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Categoria prevista", pred_cat)
            if p_cat is not None:
                st.write(f"**Confidenza categoria:** {p_cat:.2f}")
                if p_cat < CONFIDENCE_WARN:
                    st.warning("Confidenza bassa: ticket potenzialmente ambiguo (valutare revisione umana).")

        with c2:
            st.metric("Priorità suggerita (ibrida)", pred_pri)

            if pri_reason.startswith("rule"):
                if pri_reason == "rule_high":
                    st.info("Priorità determinata da **regole** (keyword critiche).")
                else:
                    st.info("Priorità determinata da **regole** (keyword di gravità media).")
            else:
                if p_pri is not None:
                    st.write(f"**Confidenza priorità (ML):** {p_pri:.2f}")
                if pri_reason == "ml_low_conf":
                    st.warning(f"Confidenza ML bassa (< {CONF_LOW:.2f}): applicata decisione conservativa.")

        st.caption(f"Motivo priorità: `{pri_reason}`")

        st.markdown("### 5 parole/frasi più influenti")
        colX, colY = st.columns(2)
        with colX:
            st.write("**Categoria**")
            _, cat_terms = top_terms(cat_model, text, k=5)
            st.dataframe(cat_terms, use_container_width=True)
        with colY:
            st.write("**Priorità**")
            _, pri_terms = top_terms(pri_model, text, k=5)
            st.dataframe(pri_terms, use_container_width=True)

        append_log({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "title": title,
            "body": body,
            "pred_category": pred_cat,
            "pred_priority": pred_pri,
            "priority_reason": pri_reason,
            "prob_category": p_cat,
            "prob_priority_ml": p_pri,
        })
        st.success("Predizione salvata nel log (data/prediction_log.csv).")


# ---------------- TAB 2: BATCH ----------------
with tab2:
    st.subheader("Predizione batch da CSV")
    st.write(
        "Carica un CSV con colonne **title** e **body**. "
        "Verrà generato un CSV con predizioni + confidenza + motivo priorità (ibrida)."
    )

    up = st.file_uploader("Carica CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)

        if not {"title", "body"}.issubset(df.columns):
            st.error("Il CSV deve contenere le colonne: title, body")
        else:
            X = (df["title"].fillna("") + " " + df["body"].fillna("")).astype(str)

            out = df.copy()

            out["pred_category"] = cat_model.predict(X)
            if hasattr(cat_model, "predict_proba"):
                out["prob_category"] = cat_model.predict_proba(X).max(axis=1)

            preds, probs, reasons = [], [], []
            for txt in X.tolist():
                p, pr, reason = predict_priority_hybrid(pri_model, txt)
                preds.append(p)
                probs.append(pr)
                reasons.append(reason)

            out["pred_priority"] = preds
            out["prob_priority_ml"] = probs
            out["priority_reason"] = reasons

            st.dataframe(out.head(30), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Scarica predizioni CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )


# ---------------- TAB 3: METRICHE ----------------
with tab3:
    st.subheader("Metriche e grafici")

    metrics_text = load_metrics_text()
    if metrics_text:
        st.code(metrics_text)

    imgs = sorted(
        glob.glob("reports/confusion_*.png")
        + glob.glob("reports/class_*.png")
        + glob.glob("reports/f1_*.png")
    )

    if imgs:
        for img in imgs:
            st.image(img, caption=os.path.basename(img), use_container_width=True)
    else:
        st.info("Nessun grafico trovato in reports/. Esegui: python -m src.train_models e poi python -m src.report_figures")

    if os.path.exists(LOG_PATH):
        st.markdown("### Log predizioni (ultime 50)")
        try:
            log_df = pd.read_csv(LOG_PATH, engine="python", on_bad_lines="skip").tail(50)
            st.dataframe(log_df, use_container_width=True)
        except Exception:
            st.warning("Log predizioni non leggibile. Puoi eliminarlo: data/prediction_log.csv (verrà ricreato).")


# ---------------- TAB 4: INFO ----------------
with tab4:
    st.subheader("Informazioni")
    st.markdown(
        """
        **STT – Smart Ticket Triage**  
        - Input: oggetto + descrizione  
        - Output: categoria (Amministrazione/Tecnico/Commerciale) e priorità (bassa/media/alta)  
        - Categoria: classificazione ML (TF-IDF + modello scelto tramite confronto LogReg vs Naive Bayes)  
        - Priorità: approccio **ibrido** (regole keyword + ML, con fallback conservativo a bassa confidenza)  
        - Spiegabilità: top-5 parole/frasi influenti calcolate sia per Logistic Regression sia per Naive Bayes  
        - Dataset: sintetico, generato ad hoc, senza dati personali.
        """
    )
    st.write("Suggerimento: per aggiornare i modelli riesegui `python -m src.train_models` e poi riavvia la dashboard.")