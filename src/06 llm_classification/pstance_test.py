"""
06_benchmark_pstance.py

Zweck:
------
Validierung des tweetbasierten Klassifikators gegen den P-Stance-Testsplit.

Die Klassifikation erfolgt mit exakt demselben Modell, Prompt und JSON-Schema
wie in classification_sample1_100_Tweets.py. Es wird nichts angepasst,
da sonst nicht die eingesetzte Pipeline, sondern eine andere gemessen wuerde.

Label-Zuordnung (P-Stance ist zielobjektbezogen und binaer):
-----------------------------------------------------------
    Donald Trump + FAVOR    ->  1  (republikanisch)
    Donald Trump + AGAINST  -> -1  (demokratisch)
    Joe Biden    + FAVOR    -> -1  (demokratisch)
    Joe Biden    + AGAINST  ->  1  (republikanisch)

P-Stance kennt keine Neutral-Klasse. Eine Ausgabe von 0 ist daher immer
falsch und wird als Verzicht auf eine Entscheidung gewertet.

Kennzahlen:
-----------
    Coverage        Anteil der Beitraege mit Ausgabe 1 oder -1
    Accuracy strikt Trefferquote ueber alle Beitraege (0 zaehlt als Fehler)
    Accuracy cov.   Trefferquote auf der Coverage-Teilmenge
    Macro-F1        Mittel aus F1(republikanisch) und F1(demokratisch),
                    0-Ausgaben zaehlen als Fehler
                    -> vergleichbar mit dem publizierten BERTweet-Wert
                       von 80,53 % (Li et al., 2021)

Quelle des Datensatzes:
-----------------------
Li, Y., Sosea, T., Sawant, A., Nair, A. J., Inkpen, D., & Caragea, C. (2021).
P-Stance: A large dataset for stance detection in political domain.
Findings of ACL-IJCNLP 2021, 2355-2365.
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

# Nur der Testsplit. Der publizierte Referenzwert bezieht sich auf ihn.
INPUT_FILES = [
    INPUT_DIR / "raw_test_trump.csv",
    INPUT_DIR / "raw_test_biden.csv",
]

MODEL = "gpt-5.4-mini"
MODEL_NAME_FOR_COLUMNS = MODEL.replace("-", "_").replace(".", "_")

OUTPUT_DIR = INPUT_DIR / "benchmark_results"
OUTPUT_FILE = OUTPUT_DIR / f"pstance_test_classified_{MODEL_NAME_FOR_COLUMNS}.csv"
CHECKPOINT_FILE = OUTPUT_DIR / f"pstance_test_checkpoint_{MODEL_NAME_FOR_COLUMNS}.csv"
METRICS_FILE = OUTPUT_DIR / f"pstance_metrics_{MODEL_NAME_FOR_COLUMNS}.csv"
CONFUSION_FILE = OUTPUT_DIR / f"pstance_confusion_{MODEL_NAME_FOR_COLUMNS}.csv"

TEXT_COLUMN = "Tweet"
TARGET_COLUMN = "Target"
STANCE_COLUMN = "Stance"

GOLD_COLUMN = "gold_alignment"
CLASSIFICATION_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_classification"
CONFIDENCE_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_confidence"

TEST_LIMIT = None
SAVE_EVERY_N_ROWS = 10

client = OpenAI(timeout=30.0, max_retries=2)


# =========================
# Prompt
# Unveraendert aus classification_sample1_100_Tweets.py uebernommen.
# =========================

def build_prompt(tweet_text: str) -> str:
    return f"""You are an expert political analyst.

Your task is to infer the overall political alignment of a single tweet contained within the <tweet> tags.

Categories of political alignment:
1 = Republican-aligned (support for Republicans or opposition to Democrats)
-1 = Democrat-aligned (support for Democrats or opposition to Republicans)
0 = Insufficient evidence, mixed, balanced, unclear, conflicting political signals, or non-political.

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
# Unveraendert aus classification_sample1_100_Tweets.py uebernommen.
# =========================

def classify_tweet(tweet_text: str, max_retries: int = 3) -> dict:
    prompt = build_prompt(tweet_text)

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


# =========================
# Sicher speichern
# =========================

def save_checkpoint(df: pd.DataFrame):
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp.csv")
    df.to_csv(temp_file, index=False, encoding="utf-8-sig")
    temp_file.replace(CHECKPOINT_FILE)


# =========================
# Kennzahlen
# =========================

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    evaluable = df[
        df[GOLD_COLUMN].notna() &
        df[CLASSIFICATION_COLUMN].notna()
    ].copy()

    evaluable[GOLD_COLUMN] = evaluable[GOLD_COLUMN].astype(int)
    evaluable[CLASSIFICATION_COLUMN] = evaluable[CLASSIFICATION_COLUMN].astype(int)

    n_total = len(evaluable)

    if n_total == 0:
        raise ValueError("Keine auswertbaren Zeilen vorhanden.")

    gold = evaluable[GOLD_COLUMN]
    pred = evaluable[CLASSIFICATION_COLUMN]

    n_neutral = int((pred == 0).sum())
    coverage = (n_total - n_neutral) / n_total

    accuracy_strict = float((gold == pred).sum()) / n_total

    covered = evaluable[pred != 0]

    if len(covered) > 0:
        accuracy_covered = float(
            (covered[GOLD_COLUMN] == covered[CLASSIFICATION_COLUMN]).sum()
        ) / len(covered)
    else:
        accuracy_covered = float("nan")

    # Macro-F1 ueber die beiden Gold-Klassen.
    # Eine Ausgabe von 0 kann nie true positive sein und geht als
    # false negative der jeweiligen Gold-Klasse ein.
    f1_scores = {}

    for label in [1, -1]:
        tp = int(((gold == label) & (pred == label)).sum())
        fp = int(((gold != label) & (pred == label)).sum())
        fn = int(((gold == label) & (pred != label)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = (f1_scores[1]["f1"] + f1_scores[-1]["f1"]) / 2

    rows = [
        {"metrik": "n_auswertbar", "wert": n_total},
        {"metrik": "n_neutral_ausgegeben", "wert": n_neutral},
        {"metrik": "coverage", "wert": round(coverage, 4)},
        {"metrik": "accuracy_strikt", "wert": round(accuracy_strict, 4)},
        {"metrik": "accuracy_coverage_teilmenge", "wert": round(accuracy_covered, 4)},
        {"metrik": "precision_republikanisch", "wert": round(f1_scores[1]["precision"], 4)},
        {"metrik": "recall_republikanisch", "wert": round(f1_scores[1]["recall"], 4)},
        {"metrik": "f1_republikanisch", "wert": round(f1_scores[1]["f1"], 4)},
        {"metrik": "precision_demokratisch", "wert": round(f1_scores[-1]["precision"], 4)},
        {"metrik": "recall_demokratisch", "wert": round(f1_scores[-1]["recall"], 4)},
        {"metrik": "f1_demokratisch", "wert": round(f1_scores[-1]["f1"], 4)},
        {"metrik": "macro_f1", "wert": round(macro_f1, 4)},
    ]

    # Zusaetzlich getrennt nach Zielobjekt, da sich die Guete
    # zwischen Trump und Biden unterscheiden koennte.
    for target in sorted(evaluable[TARGET_COLUMN].dropna().unique()):
        subset = evaluable[evaluable[TARGET_COLUMN] == target]

        acc = float(
            (subset[GOLD_COLUMN] == subset[CLASSIFICATION_COLUMN]).sum()
        ) / len(subset)

        cov = float((subset[CLASSIFICATION_COLUMN] != 0).sum()) / len(subset)

        rows.append({"metrik": f"n [{target}]", "wert": len(subset)})
        rows.append({"metrik": f"accuracy_strikt [{target}]", "wert": round(acc, 4)})
        rows.append({"metrik": f"coverage [{target}]", "wert": round(cov, 4)})

    # Mittlere Konfidenz nach Korrektheit: Grundlage fuer Abschnitt 4.3
    if CONFIDENCE_COLUMN in evaluable.columns:
        conf = pd.to_numeric(evaluable[CONFIDENCE_COLUMN], errors="coerce")
        correct_mask = (gold == pred)

        rows.append({
            "metrik": "mittlere_konfidenz_korrekt",
            "wert": round(float(conf[correct_mask].mean()), 4)
        })
        rows.append({
            "metrik": "mittlere_konfidenz_falsch",
            "wert": round(float(conf[~correct_mask].mean()), 4)
        })

    return pd.DataFrame(rows)


def compute_confusion(df: pd.DataFrame) -> pd.DataFrame:
    evaluable = df[
        df[GOLD_COLUMN].notna() &
        df[CLASSIFICATION_COLUMN].notna()
    ]

    return pd.crosstab(
        evaluable[GOLD_COLUMN].astype(int),
        evaluable[CLASSIFICATION_COLUMN].astype(int),
        rownames=["gold"],
        colnames=["vorhergesagt"],
        dropna=False,
    )


# =========================
# Hauptprogramm
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Skript gestartet.")
    print(f"Modell: {MODEL}")

    if CHECKPOINT_FILE.exists():
        print(f"Checkpoint gefunden. Lade: {CHECKPOINT_FILE}")
        df = pd.read_csv(CHECKPOINT_FILE, low_memory=False)
    else:
        print("Kein Checkpoint gefunden. Lade P-Stance-Testdateien.")

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

        # Kontrolle: Enthaelt der Datensatz wider Erwarten eine dritte Klasse?
        print("\nVorkommende Stance-Labels:")
        print(df[STANCE_COLUMN].value_counts())

        print("\nVorkommende Zielobjekte:")
        print(df[TARGET_COLUMN].value_counts())

        df[GOLD_COLUMN] = df.apply(
            lambda row: map_gold_label(row[TARGET_COLUMN], row[STANCE_COLUMN]),
            axis=1
        )

        n_unmapped = int(df[GOLD_COLUMN].isna().sum())

        if n_unmapped > 0:
            print(
                f"\nWarnung: {n_unmapped} Zeilen konnten nicht zugeordnet werden "
                f"und werden von der Auswertung ausgeschlossen."
            )

        for col in [CLASSIFICATION_COLUMN, CONFIDENCE_COLUMN]:
            df[col] = pd.NA

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    if TEST_LIMIT is not None:
        df_to_process = df.head(TEST_LIMIT)
        print(f"\nTestmodus aktiv: maximal {TEST_LIMIT} Beitraege.")
    else:
        df_to_process = df
        print("\nTestmodus deaktiviert: alle Beitraege werden verarbeitet.")

    remaining = df_to_process[
        df_to_process[GOLD_COLUMN].notna() &
        (
            df_to_process[CLASSIFICATION_COLUMN].isna() |
            df_to_process[CONFIDENCE_COLUMN].isna()
        )
    ]

    print(f"Bereits klassifiziert: {len(df_to_process) - len(remaining)}")
    print(f"Noch offen: {len(remaining)}")

    for counter, (idx, row) in enumerate(
        tqdm(
            remaining.iterrows(),
            total=len(remaining),
            desc=f"P-Stance klassifizieren mit {MODEL}"
        ),
        start=1
    ):
        result = classify_tweet(str(row[TEXT_COLUMN]))

        df.at[idx, CLASSIFICATION_COLUMN] = result["classification"]
        df.at[idx, CONFIDENCE_COLUMN] = result["confidence"]

        if counter % SAVE_EVERY_N_ROWS == 0:
            save_checkpoint(df)

    save_checkpoint(df)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # =========================
    # Auswertung
    # =========================

    metrics_df = compute_metrics(df)
    metrics_df.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")

    confusion_df = compute_confusion(df)
    confusion_df.to_csv(CONFUSION_FILE, encoding="utf-8-sig")

    print("\n=== Kennzahlen ===")
    print(metrics_df.to_string(index=False))

    print("\n=== Konfusionsmatrix ===")
    print(confusion_df)

    print("\nFertig.")
    print(f"Klassifikationen: {OUTPUT_FILE}")
    print(f"Kennzahlen: {METRICS_FILE}")
    print(f"Konfusionsmatrix: {CONFUSION_FILE}")


if __name__ == "__main__":
    main()