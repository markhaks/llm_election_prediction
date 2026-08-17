"""
07_prompt_order_test_val.py

Zweck:
------
Diagnose eines moeglichen Positionseffekts der Label-Reihenfolge im Prompt.

Die Auswertung des P-Stance-Testsplits ergab eine asymmetrische Fehlerverteilung
zulasten demokratisch ausgerichteter Beitraege. Da die republikanische Kategorie
im eingesetzten Prompt an erster Stelle steht, ist nicht auszuschliessen, dass die
Verzerrung nicht inhaltlich bedingt, sondern ein Artefakt der Prompt-Gestaltung ist.

Vorgehen:
---------
Zwei Prompt-Varianten werden auf denselben Zeilen des Validierungssplits
ausgefuehrt. Beide unterscheiden sich ausschliesslich in der Reihenfolge der
aufgefuehrten Kategorien; Wortlaut, Zahlencodes, Systemnachricht, Schema und
Modellparameter bleiben identisch.

    Variante A: 1 (republikanisch), -1 (demokratisch), 0   [Originalprompt]
    Variante B: -1 (demokratisch), 1 (republikanisch), 0   [getauscht]

Interpretation:
---------------
    Verzerrung dreht sich um  -> Positionseffekt des Prompts.
                                 Der Befund des Testsplits ist als Artefakt
                                 einzuordnen und in Kapitel 5 zu relativieren.
    Verzerrung bleibt         -> Das Modell erkennt demokratische Positionierung
                                 tatsaechlich schlechter. Der Befund ist robust.

Wichtig:
--------
Es wird ausschliesslich der Validierungssplit verwendet. Der Testsplit bleibt fuer
die in Abschnitt 3.6.2 berichtete Kennzahl reserviert, da dessen Ergebnisse bereits
gesichtet wurden und eine erneute Messung darauf auf den Testsatz optimieren wuerde.

Dieses Skript aendert nicht den Prompt der Hauptstudie. Es dient allein der
Einordnung der Fehlerverteilung.
"""

from pathlib import Path
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# =========================
# Einstellungen
# =========================

INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\external\pstance"
)

INPUT_FILES = [
    INPUT_DIR / "raw_val_trump.csv",
    INPUT_DIR / "raw_val_biden.csv",
]

MODEL = "gpt-5.4-mini"
MODEL_NAME_FOR_COLUMNS = MODEL.replace("-", "_").replace(".", "_")

OUTPUT_DIR = INPUT_DIR / "prompt_order_test"
OUTPUT_FILE = OUTPUT_DIR / f"val_prompt_order_classified_{MODEL_NAME_FOR_COLUMNS}.csv"
CHECKPOINT_FILE = OUTPUT_DIR / f"val_prompt_order_checkpoint_{MODEL_NAME_FOR_COLUMNS}.csv"
METRICS_FILE = OUTPUT_DIR / f"val_prompt_order_metrics_{MODEL_NAME_FOR_COLUMNS}.csv"
CONFUSION_FILE = OUTPUT_DIR / f"val_prompt_order_confusion_{MODEL_NAME_FOR_COLUMNS}.csv"

TEXT_COLUMN = "Tweet"
TARGET_COLUMN = "Target"
STANCE_COLUMN = "Stance"

GOLD_COLUMN = "gold_alignment"

# Beide Varianten werden auf denselben Zeilen ausgefuehrt.
VARIANTS = ["A", "B"]

# Kostenbegrenzung: Zeilen je Zielobjekt. None = vollstaendiger Split.
# 400 je Zielobjekt genuegen, um eine gerichtete Verschiebung zu erkennen.
SAMPLE_PER_TARGET = 400
RANDOM_SEED = 42

SAVE_EVERY_N_ROWS = 10

client = OpenAI(timeout=30.0, max_retries=2)


# =========================
# Prompts
# Beide Varianten sind wortgleich. Abweichend ist ausschliesslich die
# Reihenfolge der drei Kategoriezeilen.
# =========================

CATEGORIES_A = """1 = Republican-aligned (support for Republicans or opposition to Democrats)
-1 = Democrat-aligned (support for Democrats or opposition to Republicans)
0 = Insufficient evidence, mixed, balanced, unclear, conflicting political signals, or non-political."""

CATEGORIES_B = """-1 = Democrat-aligned (support for Democrats or opposition to Republicans)
1 = Republican-aligned (support for Republicans or opposition to Democrats)
0 = Insufficient evidence, mixed, balanced, unclear, conflicting political signals, or non-political."""

CATEGORIES = {
    "A": CATEGORIES_A,
    "B": CATEGORIES_B,
}


def build_prompt(tweet_text: str, variant: str) -> str:
    return f"""You are an expert political analyst.

Your task is to infer the overall political alignment of a single tweet contained within the <tweet> tags.

Categories of political alignment:
{CATEGORIES[variant]}

Confidence:
Return a confidence score between 0.0 (completely uncertain) and 1.0 (completely certain). The confidence should reflect your certainty in the tweet overall political alignment.

Output requirements:
- Return exactly one political alignment for the tweet.
- Return ONLY a valid JSON object.
- Do not include explanations, markdown, or additional text.
- The JSON object must contain exactly these two keys:
  - "classification"
  - "confidence"

Desired JSON structure:
{{"classification": <integer>, "confidence": <float>}}

<tweet>
{tweet_text}
</tweet>"""


# =========================
# API-Klassifikation
# Modellparameter und Schema unveraendert aus der Hauptstudie.
# =========================

def classify_tweet(tweet_text: str, variant: str, max_retries: int = 3) -> dict:
    prompt = build_prompt(tweet_text, variant)

    json_schema = {
        "name": "tweet_classification_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "integer",
                    "enum": [-1, 0, 1]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["classification", "confidence"],
            "additionalProperties": False
        }
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                },
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert political analyst specializing in US politics."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            result = json.loads(response.choices[0].message.content)

            return {
                "classification": int(result["classification"]),
                "confidence": float(result["confidence"])
            }

        except Exception as e:
            wait_time = 2 ** attempt
            print(f"Fehler bei API-Aufruf, Versuch {attempt + 1}/{max_retries}: {e}")
            time.sleep(wait_time)

    return {
        "classification": None,
        "confidence": None
    }


# =========================
# Label-Zuordnung
# =========================

GOLD_MAPPING = {
    ("donald trump", "favor"): 1,
    ("donald trump", "against"): -1,
    ("joe biden", "favor"): -1,
    ("joe biden", "against"): 1,
}


def map_gold_label(target, stance):
    key = (str(target).strip().lower(), str(stance).strip().lower())
    return GOLD_MAPPING.get(key)


def cls_col(variant: str) -> str:
    return f"{MODEL_NAME_FOR_COLUMNS}_classification_{variant}"


def conf_col(variant: str) -> str:
    return f"{MODEL_NAME_FOR_COLUMNS}_confidence_{variant}"


# =========================
# Sicher speichern
# =========================

def save_checkpoint(df: pd.DataFrame):
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp.csv")
    df.to_csv(temp_file, index=False, encoding="utf-8-sig")
    temp_file.replace(CHECKPOINT_FILE)


# =========================
# Kennzahlen je Variante
# =========================

def variant_metrics(df: pd.DataFrame, variant: str) -> dict:
    column = cls_col(variant)

    evaluable = df[
        df[GOLD_COLUMN].notna() &
        df[column].notna()
    ].copy()

    gold = evaluable[GOLD_COLUMN].astype(int)
    pred = evaluable[column].astype(int)

    n_total = len(evaluable)

    metrics = {"n_auswertbar": n_total}

    metrics["n_neutral_ausgegeben"] = int((pred == 0).sum())
    metrics["coverage"] = round((pred != 0).sum() / n_total, 4)
    metrics["accuracy_strikt"] = round(float((gold == pred).sum()) / n_total, 4)

    covered = pred != 0

    if covered.sum() > 0:
        metrics["accuracy_coverage_teilmenge"] = round(
            float((gold[covered] == pred[covered]).sum()) / int(covered.sum()), 4
        )
    else:
        metrics["accuracy_coverage_teilmenge"] = float("nan")

    f1_values = {}

    for label, name in [(1, "republikanisch"), (-1, "demokratisch")]:
        tp = int(((gold == label) & (pred == label)).sum())
        fp = int(((gold != label) & (pred == label)).sum())
        fn = int(((gold == label) & (pred != label)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_values[label] = f1

        metrics[f"precision_{name}"] = round(precision, 4)
        metrics[f"recall_{name}"] = round(recall, 4)
        metrics[f"f1_{name}"] = round(f1, 4)

    metrics["macro_f1"] = round((f1_values[1] + f1_values[-1]) / 2, 4)

    # Zentrale Groesse: republikanischer Anteil unter den festgelegten Ausgaben
    # gegenueber dem republikanischen Anteil im Gold-Standard.
    n_pred_rep = int((pred == 1).sum())
    n_pred_dem = int((pred == -1).sum())

    if (n_pred_rep + n_pred_dem) > 0:
        share_pred = n_pred_rep / (n_pred_rep + n_pred_dem)
    else:
        share_pred = float("nan")

    n_gold_rep = int((gold == 1).sum())
    n_gold_dem = int((gold == -1).sum())
    share_gold = n_gold_rep / (n_gold_rep + n_gold_dem)

    metrics["anteil_republikanisch_gold"] = round(share_gold, 4)
    metrics["anteil_republikanisch_vorhergesagt"] = round(share_pred, 4)
    metrics["verzerrung_anteil_pp"] = round((share_pred - share_gold) * 100, 2)
    metrics["verzerrung_gewinnspanne_pp"] = round((share_pred - share_gold) * 200, 2)

    conf = pd.to_numeric(evaluable[conf_col(variant)], errors="coerce")
    correct = (gold == pred)

    metrics["mittlere_konfidenz_korrekt"] = round(float(conf[correct].mean()), 4)
    metrics["mittlere_konfidenz_falsch"] = round(float(conf[~correct].mean()), 4)

    return metrics


def build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}

    for variant in VARIANTS:
        rows[f"Variante_{variant}"] = variant_metrics(df, variant)

    comparison = pd.DataFrame(rows)
    comparison.index.name = "metrik"

    return comparison.reset_index()


def build_agreement(df: pd.DataFrame) -> pd.DataFrame:
    both = df[
        df[cls_col("A")].notna() &
        df[cls_col("B")].notna()
    ]

    return pd.crosstab(
        both[cls_col("A")].astype(int),
        both[cls_col("B")].astype(int),
        rownames=["Variante A"],
        colnames=["Variante B"],
        dropna=False,
    )


# =========================
# Hauptprogramm
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Skript gestartet.")
    print(f"Modell: {MODEL}")
    print("Hinweis: Es wird ausschliesslich der Validierungssplit verwendet.")

    if CHECKPOINT_FILE.exists():
        print(f"Checkpoint gefunden. Lade: {CHECKPOINT_FILE}")
        df = pd.read_csv(CHECKPOINT_FILE, low_memory=False)
    else:
        print("Kein Checkpoint gefunden. Lade Validierungsdateien.")

        frames = []

        for file_path in INPUT_FILES:
            if not file_path.exists():
                raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

            temp = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
            temp["source_file"] = file_path.name
            frames.append(temp)

            print(f"Geladen: {file_path.name} ({len(temp):,} Zeilen)")

        df = pd.concat(frames, ignore_index=True)

        required = [TEXT_COLUMN, TARGET_COLUMN, STANCE_COLUMN]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Fehlende Spalten: {missing}")

        df[GOLD_COLUMN] = df.apply(
            lambda row: map_gold_label(row[TARGET_COLUMN], row[STANCE_COLUMN]),
            axis=1
        )

        n_unmapped = int(df[GOLD_COLUMN].isna().sum())

        if n_unmapped > 0:
            print(f"Warnung: {n_unmapped} Zeilen nicht zugeordnet, werden ausgeschlossen.")

        df = df[df[GOLD_COLUMN].notna()].copy()

        if SAMPLE_PER_TARGET is not None:
            df = (
                df
                .groupby(TARGET_COLUMN, group_keys=False)
                .apply(
                    lambda g: g.sample(
                        n=min(SAMPLE_PER_TARGET, len(g)),
                        random_state=RANDOM_SEED
                    )
                )
                .reset_index(drop=True)
            )

            print(
                f"Teilstichprobe: maximal {SAMPLE_PER_TARGET} Zeilen je Zielobjekt "
                f"(Startwert {RANDOM_SEED}). Verbleibend: {len(df):,} Zeilen."
            )

        for variant in VARIANTS:
            df[cls_col(variant)] = pd.NA
            df[conf_col(variant)] = pd.NA

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    n_calls = sum(
        int(df[cls_col(v)].isna().sum() | df[conf_col(v)].isna().sum())
        for v in VARIANTS
    )

    print(f"\nZeilen: {len(df):,}")
    print(f"Offene API-Aufrufe (beide Varianten): rund {n_calls:,}")

    for variant in VARIANTS:
        column = cls_col(variant)
        confidence = conf_col(variant)

        remaining = df[df[column].isna() | df[confidence].isna()]

        print(f"\nVariante {variant}: {len(remaining):,} offene Zeilen.")

        if len(remaining) == 0:
            continue

        for counter, (idx, row) in enumerate(
            tqdm(
                remaining.iterrows(),
                total=len(remaining),
                desc=f"Variante {variant}"
            ),
            start=1
        ):
            result = classify_tweet(str(row[TEXT_COLUMN]), variant)

            df.at[idx, column] = result["classification"]
            df.at[idx, confidence] = result["confidence"]

            if counter % SAVE_EVERY_N_ROWS == 0:
                save_checkpoint(df)

        save_checkpoint(df)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # =========================
    # Auswertung
    # =========================

    comparison_df = build_comparison(df)
    comparison_df.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")

    agreement_df = build_agreement(df)
    agreement_df.to_csv(CONFUSION_FILE, encoding="utf-8-sig")

    print("\n=== Vergleich der Prompt-Varianten ===")
    print(comparison_df.to_string(index=False))

    print("\n=== Uebereinstimmung der Varianten ===")
    print(agreement_df)

    both = df[df[cls_col("A")].notna() & df[cls_col("B")].notna()]

    if len(both) > 0:
        identical = int(
            (both[cls_col("A")].astype(int) == both[cls_col("B")].astype(int)).sum()
        )

        print(
            f"\nIdentische Ausgaben: {identical:,} von {len(both):,} "
            f"({identical / len(both) * 100:.2f} %)"
        )

    bias_a = comparison_df.loc[
        comparison_df["metrik"] == "verzerrung_anteil_pp", "Variante_A"
    ].iloc[0]

    bias_b = comparison_df.loc[
        comparison_df["metrik"] == "verzerrung_anteil_pp", "Variante_B"
    ].iloc[0]

    print("\n=== Einordnung ===")
    print(f"Verzerrung Variante A (Original): {bias_a:+.2f} Prozentpunkte")
    print(f"Verzerrung Variante B (getauscht): {bias_b:+.2f} Prozentpunkte")
    print("Positive Werte kennzeichnen eine Verschiebung zugunsten der Republikaner.")

    if bias_a * bias_b < 0:
        print(
            "\nDie Verzerrung wechselt das Vorzeichen. Dies deutet auf einen "
            "Positionseffekt der Label-Reihenfolge hin."
        )
    else:
        print(
            "\nDie Verzerrung behaelt ihr Vorzeichen. Ein Positionseffekt als "
            "alleinige Ursache erscheint damit unwahrscheinlich."
        )

    print("\nFertig.")
    print(f"Klassifikationen: {OUTPUT_FILE}")
    print(f"Vergleich: {METRICS_FILE}")
    print(f"Uebereinstimmung: {CONFUSION_FILE}")


if __name__ == "__main__":
    main()