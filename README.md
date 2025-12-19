# PW18 – Triage automatico ticket di assistenza

Project Work – Laurea Triennale in Informatica per le Aziende Digitali  
Traccia 18 – Classificazione automatica ticket di assistenza

---

## Obiettivo del progetto

Il progetto realizza un **sistema di triage automatico** per ticket di assistenza clienti che, dato un testo composto da **oggetto** e **descrizione**, è in grado di:

- classificare il ticket in una **categoria**:
  - Amministrazione
  - Tecnico
  - Commerciale
- stimare la **priorità** del ticket:
  - bassa
  - media
  - alta
- fornire una **spiegazione** tramite le parole/frasi più influenti
- supportare **predizioni batch da CSV**
- visualizzare risultati e metriche tramite **dashboard web**

Il progetto utilizza **dataset sintetico**, generato ad hoc, e **non contiene dati personali**.

---

## Struttura del progetto
```bash
PW 18/
│
├── app/
│   └── streamlit_app.py        # Dashboard Streamlit
│
├── data/
│   ├── tickets.csv             # Dataset sintetico
│   ├── predictions.csv         # Output batch
│   └── prediction_log.csv      # Log dashboard
│
├── models/
│   ├── category_model.joblib
│   └── priority_model.joblib
│
├── reports/
│   ├── confusion_*.png
│   ├── class_distribution_*.png
│   ├── f1_per_class_*.png
│   ├── metrics_summary.txt
│   └── metrics.txt
│
├── src/
│   ├── __init__.py
│   ├── explain.py              # Spiegabilità (top-words LogReg + NB)
│   ├── features.py             # Preprocessing testo
│   ├── generate_dataset.py     # Generazione dataset sintetico
│   ├── predict_batch.py        # Predizione batch CSV
│   ├── priority_hybrid.py      # Priorità ibrida (regole + ML)
│   └── report_figures.py       # Grafici per il report
│   ├── train_models.py         # Training e valutazione modelli
│
├── requirements.txt
└── README.md
```
---

## Requisiti

- Python **3.10+**
- Sistema operativo: Windows / Linux / macOS

Librerie principali:
- pandas
- scikit-learn
- matplotlib
- joblib
- streamlit

---

## Installazione e dipendenze

Il progetto utilizza un **ambiente virtuale Python** per garantire isolamento e riproducibilità.

## Creazione ambiente virtuale

```bash
python -m venv venv
```

## Attivazione ambiente

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

## Installazione librerie

Tutte le dipendenze sono elencate nel file `requirements.txt`.

```bash
pip install -r requirements.txt
```

Il file `requirements.txt` include:

* `pandas` – gestione dati e CSV
* `scikit-learn` – modelli di Machine Learning
* `matplotlib` – grafici e confusion matrix
* `joblib` – salvataggio/caricamento modelli
* `streamlit` – dashboard web interattiva

---

## Generazione dataset sintetico

Il dataset viene generato automaticamente con ticket realistici.

```bash
python -m src.generate_dataset --n 350 --out data/tickets.csv
```

Colonne generate:

* `id`
* `title`
* `body`
* `category`
* `priority`

✔️ Requisito traccia: **dataset sintetico 200–500 ticket**

---

## Training e valutazione modelli

```bash
python -m src.train_models > reports/metrics.txt
```

Durante il training:

* split **80% training / 20% test**
* confronto **Logistic Regression vs Naive Bayes** per la categoria
* selezione automatica del modello migliore (F1 macro)
* training modello priorità (Logistic Regression)

Metriche calcolate:

* Accuracy
* F1 macro
* F1 per classe
* Confusion Matrix

✔️ Requisito traccia: **valutazione modelli**

---

## Grafici per il report

```bash
python -m src.report_figures
```

Grafici prodotti:

* Distribuzione classi (categoria e priorità)
* F1-score per classe
* Confusion matrix

✔️ Requisito traccia: **grafici e analisi risultati**

---

## Predizione batch da CSV

```bash
python -m src.predict_batch
```

Input:

* `data/tickets.csv` oppure CSV personalizzato con colonne `title`, `body`

Output:

* `data/predictions.csv` con:

  * categoria predetta
  * priorità predetta
  * probabilità
  * motivo priorità (regole / ML)

✔️ Requisito traccia: **batch di ticket**

---

## Dashboard interattiva

```bash
streamlit run app/streamlit_app.py
```

Funzionalità:

* Inserimento ticket singolo
* Classificazione categoria e priorità
* Priorità **ibrida** (regole + ML)
* Visualizzazione **top-5 parole influenti**
* Upload CSV batch
* Visualizzazione metriche e grafici
* Log automatico delle predizioni

✔️ Requisito traccia: **interfaccia grafica**

---

## Priorità ibrida

La priorità è stimata con approccio **ibrido**:

1. Regole basate su keyword critiche (es. *bloccante*, *crash*, *errore 500*)
2. Modello ML per casi non critici
3. Fallback conservativo in caso di bassa confidenza

✔️ Miglioramento realistico “da contesto aziendale”

---

## Spiegabilità del modello

Per ogni predizione vengono mostrate le **5 parole/frasi più influenti**, calcolate:

* per **Logistic Regression** tramite coefficienti
* per **Naive Bayes** tramite probabilità logaritmiche

✔️ Requisito traccia: **interpretabilità**

---

## Allineamento con la Traccia 18

| Requisito traccia          | Stato |
| -------------------------- | ----- |
| Dataset sintetico 200–500  | ✅    |
| Classificazione categoria  | ✅    |
| Stima priorità             | ✅    |
| Preprocessing testo        | ✅    |
| Modelli ML                 | ✅    |
| Valutazione (Accuracy, F1) | ✅    |
| Confusion Matrix           | ✅    |
| Batch CSV                  | ✅    |
| Dashboard grafica          | ✅    |
| Spiegabilità               | ✅    |

## 👤 Autore

Project Work realizzato da **Giancarlo Ierardi - Matr 0312300194**
Corso di Laurea in Informatica per le Aziende Digitali

---

## Reset del progetto (pulizia completa)

Questa sezione permette di **ripulire completamente il progetto** eliminando file generati automaticamente (dataset, modelli, report), così da poter **rigenerare tutto da zero** in modo riproducibile.

## File e cartelle generati automaticamente

I seguenti elementi **non fanno parte del codice sorgente** e vengono creati durante l’esecuzione:

* `data/*.csv` → dataset e predizioni
* `models/*.joblib` → modelli addestrati
* `reports/*.png` → grafici e confusion matrix
* `reports/*.txt` → metriche
* `data/prediction_log.csv` → log dashboard
* `__pycache__/` → cache Python

---

## Pulizia manuale (CMD – Windows)

Eseguire i seguenti comandi **dalla root del progetto**.

## Eliminare dataset e output

```bash
del /Q data\*.csv
```

## Eliminare modelli addestrati

```bash
del /Q models\*.joblib
```

## Eliminare report e metriche

```bash
del /Q reports\*.png
del /Q reports\*.txt
```

### Eliminare log predizioni dashboard

```bash
del /Q data\prediction_log.csv
```

### Eliminare cache Python

```bash
rmdir /S /Q src\__pycache__
rmdir /S /Q app\__pycache__
```

---

## Reset completo (opzionale)

Per una pulizia totale, inclusa la rimozione dell’ambiente virtuale:

```bash
rmdir /S /Q venv
```

Dopo questo comando sarà necessario ricreare l’ambiente virtuale e reinstallare le librerie.

---

## Rigenerazione completa da zero

Dopo la pulizia, per rigenerare l’intero progetto:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_dataset --n 350 --out data\tickets.csv
python -m src.train_models > reports\metrics.txt
python -m src.report_figures
python -m src.predict_batch
streamlit run app\streamlit_app.py
```

---
