# PW18 – Triage automatico ticket di assistenza

Project Work – Laurea Triennale in Informatica per le Aziende Digitali  
Traccia 18 – Classificazione automatica ticket di assistenza

---

## Obiettivo del progetto

Il progetto sviluppa un **sistema di triage automatico** per i ticket di assistenza clienti che, partendo da un testo composto da **oggetto** e **descrizione**, è capace di:

- classificare il ticket in una **categoria**:
  - Amministrazione
  - Tecnico
  - Commerciale
- stimare la **priorità** del ticket:
  - bassa
  - media
  - alta
- fornire una **spiegazione** utilizzando le parole o frasi più significative
- supportare **predizioni batch da CSV**
- visualizzare risultati e metriche attraverso una **dashboard web**

Il progetto si avvale di un **dataset sintetico**, creato appositamente, e **non include dati personali**.

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

Il progetto sfrutta un **ambiente virtuale Python** per assicurare isolamento e riproducibilità.

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

Tutte le dipendenze sono indicate nel file `requirements.txt`.

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

Il dataset viene creato automaticamente con ticket che sembrano realistici.

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

* abbiamo diviso i dati in **80% per il training e 20% per il test**
* abbiamo confrontato **Logistic Regression e Naive Bayes** per la categoria
* abbiamo selezionato automaticamente il modello migliore basandoci sull'F1 macro
* abbiamo dato priorità al training del modello **Logistic Regression**

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

* Inserimento di un ticket singolo
* Classificazione per categoria e priorità
* Priorità **ibrida** (regole + machine learning)
* Visualizzazione delle **top-5 parole influenti**
* Upload di file CSV in batch
* Visualizzazione di metriche e grafici
* Log automatico delle predizioni

✔️ Requisito traccia: **interfaccia grafica**

---

## Priorità ibrida

La priorità viene stimata attraverso un approccio **ibrido**:

1. Regole basate su parole chiave critiche (ad esempio *bloccante*, *crash*, *errore 500*)
2. Modello di machine learning per casi non critici
3. Fallback conservativo in caso di bassa confidenza

✔️ Miglioramento realistico “da contesto aziendale”

---

## Spiegabilità del modello

Per ogni predizione, vengono mostrate le **5 parole/frasi più influenti**, calcolate:

* per **Logistic Regression** tramite coefficienti
* per **Naive Bayes** tramite probabilità logaritmiche

✔️ Requisito traccia: **interpretabilità**

---

## 👤 Autore

Project Work realizzato da **Giancarlo Ierardi - Matr 0312300194**
Corso di Laurea in Informatica per le Aziende Digitali

---

## Reset del progetto (pulizia completa)

Questa sezione consente di **ripulire completamente il progetto**, rimuovendo file generati automaticamente come dataset, modelli e report. In questo modo, puoi **rigenerare tutto da zero** in modo riproducibile.

## File e cartelle generati automaticamente

I seguenti elementi **non fanno parte del codice sorgente** e vengono creati durante l'esecuzione:

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
