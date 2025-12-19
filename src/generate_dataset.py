from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


CATEGORIES = ["Amministrazione", "Tecnico", "Commerciale"]


@dataclass
class TicketTemplate:
    title_patterns: List[str]
    body_patterns: List[str]
    keywords: List[str]


TEMPLATES: Dict[str, TicketTemplate] = {
    "Amministrazione": TicketTemplate(
        title_patterns=[
            "Problema fattura {n}", "Richiesta copia fattura", "Pagamento non registrato",
            "Nota di credito", "IBAN per bonifico", "Scadenza pagamento {n}"
        ],
        body_patterns=[
            "Buongiorno, la fattura {n} risulta {issue}. Potete verificare?",
            "Ho bisogno della copia della fattura {n} per la contabilità.",
            "Il pagamento del {date} non appare ancora, è {impact}.",
            "Vorrei sapere come procedere per {action} e tempi di emissione."
        ],
        keywords=["fattura", "pagamento", "bonifico", "iban", "nota di credito", "scadenza", "ricevuta"]
    ),
    "Tecnico": TicketTemplate(
        title_patterns=[
            "Errore 500 su login", "Servizio non disponibile", "Bug app mobile", "Problema accesso",
            "Crash dopo aggiornamento", "Prestazioni lente", "API non risponde"
        ],
        body_patterns=[
            "Da stamattina vedo l'errore '{err}' quando provo a {action}. È {impact}.",
            "Il sistema sembra {issue} su {area}. Potete intervenire? È {impact}.",
            "Dopo l'aggiornamento {ver} l'app va in crash, problema {impact}.",
            "Non riesco ad accedere: {issue}. Ho già provato a {workaround}."
        ],
        keywords=["errore", "bug", "crash", "login", "api", "timeout", "non disponibile", "bloccante", "urgente"]
    ),
    "Commerciale": TicketTemplate(
        title_patterns=[
            "Richiesta preventivo", "Informazioni su offerta", "Cambio piano", "Sconto per rinnovo",
            "Ordine {n} in corso", "Tempi di consegna", "Dettagli funzionalità"
        ],
        body_patterns=[
            "Vorrei un preventivo per {qty} licenze con opzione {opt}.",
            "Potete darmi informazioni sulla vostra offerta {plan} e costi?",
            "Desidero cambiare piano da {plan_old} a {plan_new} dal prossimo mese.",
            "Ho un ordine {n}: qual è lo stato e tempi stimati?"
        ],
        keywords=["preventivo", "offerta", "piano", "sconto", "ordine", "rinnovo", "demo"]
    ),
}

ISSUES = ["non corretta", "duplicata", "mancante", "in ritardo", "errata"]
IMPACTS = ["bloccante", "urgente", "gestibile", "non critico", "critico"]
AREAS = ["dashboard", "portale", "checkout", "area clienti", "reportistica"]
ERRORS = ["HTTP 500", "timeout", "403 Forbidden", "connessione rifiutata", "token scaduto"]
WORKAROUNDS = ["svuotare cache", "cambiare browser", "resettare password", "riavviare l'app"]
PLANS = ["Base", "Pro", "Business", "Enterprise"]
OPTS = ["assistenza premium", "SSO", "backup avanzato", "reportistica"]
DATES = ["2025-10-02", "2025-10-11", "2025-11-03", "2025-11-18", "2025-12-05"]

NOISE_WORDS = [
    "gentilmente", "tempistiche", "cliente", "ticket", "allego", "screen", "grazie",
    "cordiali saluti", "ASAP", "follow-up", "impatto", "SLA", "priorità", "helpdesk"
]

SYNONYMS = {
    "errore": ["problema", "anomalia", "malfunzionamento"],
    "pagamento": ["versamento", "saldo"],
    "fattura": ["invoice", "documento fiscale"],
    "preventivo": ["quotazione", "offerta economica"],
    "crash": ["si chiude", "si blocca", "va in crash"],
    "login": ["accesso", "signin"],
    "ordine": ["acquisto", "purchase"],
    "lente": ["rallentata", "slow"],
}


def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def infer_priority(text: str) -> str:
    t = text.lower()
    high = ["bloccante", "urgente", "critico", "non disponibile", "crash", "errore 500", "api non risponde"]
    medium = ["in ritardo", "lente", "timeout", "mancante", "non corretta", "rallent"]
    if any(k in t for k in high):
        return "alta"
    if any(k in t for k in medium):
        return "media"
    return "bassa"


def replace_synonyms(s: str, p: float) -> str:
    if p <= 0:
        return s
    t = s
    low = t.lower()
    for key, vals in SYNONYMS.items():
        if random.random() < p and re.search(rf"\b{re.escape(key)}\b", low):
            t = re.sub(rf"\b{re.escape(key)}\b", random.choice(vals), t, flags=re.IGNORECASE)
            low = t.lower()
    return t


def add_typos(s: str, p: float) -> str:
    if p <= 0:
        return s
    words = s.split()
    out = []
    for w in words:
        if len(w) >= 6 and random.random() < p:
            i = random.randint(1, len(w) - 2)
            if random.random() < 0.5:
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def add_noise_words(s: str, noise: float) -> str:
    if noise <= 0:
        return s
    if random.random() < noise:
        k = random.randint(1, 3)
        s = clean_spaces(s + " " + " ".join(random.sample(NOISE_WORDS, k=k)))
    return s


def make_mixed_body(main_cat: str) -> str:
    other = random.choice([c for c in CATEGORIES if c != main_cat])
    other_tpl = TEMPLATES[other]

    extra = random.choice(other_tpl.body_patterns).format(
        n=random.randint(1000, 9999),
        issue=random.choice(ISSUES),
        impact=random.choice(IMPACTS),
        action=random.choice(["accedere", "emettere una nota di credito", "verificare lo stato"]),
        date=random.choice(DATES),
        area=random.choice(AREAS),
        err=random.choice(ERRORS),
        workaround=random.choice(WORKAROUNDS),
        ver=f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}",
        qty=random.choice([3, 5, 10, 25, 50]),
        opt=random.choice(OPTS),
        plan=random.choice(PLANS),
        plan_old=random.choice(PLANS),
        plan_new=random.choice(PLANS),
    )

    if random.random() < 0.70:
        extra += " " + random.choice(other_tpl.keywords)

    return clean_spaces(extra)


def make_one(category: str, i: int, noise: float, mix: float, label_noise: float) -> dict:
    tpl = TEMPLATES[category]

    title = random.choice(tpl.title_patterns).format(n=random.randint(1000, 9999))
    body = random.choice(tpl.body_patterns).format(
        n=random.randint(1000, 9999),
        issue=random.choice(ISSUES),
        impact=random.choice(IMPACTS),
        action=random.choice(["accedere", "emettere una nota di credito", "verificare lo stato"]),
        date=random.choice(DATES),
        area=random.choice(AREAS),
        err=random.choice(ERRORS),
        workaround=random.choice(WORKAROUNDS),
        ver=f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}",
        qty=random.choice([3, 5, 10, 25, 50]),
        opt=random.choice(OPTS),
        plan=random.choice(PLANS),
        plan_old=random.choice(PLANS),
        plan_new=random.choice(PLANS),
    )

    if random.random() < 0.70:
        body = clean_spaces(body + " " + random.choice(tpl.keywords))

    if mix > 0 and random.random() < mix:
        body = clean_spaces(body + " Inoltre: " + make_mixed_body(category))

    title = add_noise_words(title, noise)
    body = add_noise_words(body, noise)

    text = clean_spaces(title + " " + body)
    priority = infer_priority(text)

    out_category = category
    if label_noise > 0 and random.random() < label_noise:
        out_category = random.choice([c for c in CATEGORIES if c != category])

    return {
        "id": i,
        "title": clean_spaces(title),
        "body": clean_spaces(body),
        "category": out_category,
        "priority": priority,
    }


def generate(n: int, noise: float, mix: float, label_noise: float) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        cat = random.choices(CATEGORIES, weights=[0.34, 0.33, 0.33])[0]
        rows.append(make_one(cat, i, noise=noise, mix=mix, label_noise=label_noise))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None, help="Seed RNG. Se non fornito, dataset diverso ad ogni run.")
    p.add_argument("--n", type=int, default=350)
    p.add_argument("--out", type=str, default="data/tickets.csv")

    p.add_argument("--noise", type=float, default=0.15, help="Rumore testuale (0-0.30)")
    p.add_argument("--mix", type=float, default=0.25, help="Probabilità ticket multi-intento (0-0.40)")
    p.add_argument("--label-noise", type=float, default=0.07, help="Probabilità etichetta categoria errata (0-0.15)")
    p.add_argument("--typo", type=float, default=0.04, help="Probabilità typo per parola (0-0.08)")
    p.add_argument("--syn", type=float, default=0.25, help="Probabilità sostituzione sinonimi (0-0.50)")

    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    df = generate(args.n, noise=args.noise, mix=args.mix, label_noise=args.label_noise)

    df["title"] = df["title"].apply(lambda s: add_typos(replace_synonyms(s, args.syn), args.typo))
    df["body"] = df["body"].apply(lambda s: add_typos(replace_synonyms(s, args.syn), args.typo))

    df.to_csv(args.out, index=False)
    print(f"Creato dataset: {args.out} ({len(df)} righe)")
    print(
        f"Parametri: seed={args.seed} noise={args.noise} mix={args.mix} "
        f"label_noise={args.label_noise} typo={args.typo} syn={args.syn}"
    )


if __name__ == "__main__":
    main()
