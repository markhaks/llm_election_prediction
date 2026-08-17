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
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_5 Final samples\sample1"
)

INPUT_FILE = INPUT_DIR / "sample_400_tweets_per_state.csv"



MODEL = "gpt-5.4-mini"
MODEL_NAME_FOR_COLUMNS = MODEL.replace("-", "_").replace(".", "_")

OUTPUT_FILE = INPUT_DIR / f"sample_400_tweets_per_state_llm_classified_{MODEL_NAME_FOR_COLUMNS}.csv"

CHECKPOINT_FILE = INPUT_DIR / f"sample_400_tweets_per_state_llm_checkpoint_{MODEL_NAME_FOR_COLUMNS}.csv"

TEXT_COLUMN = "tweet_text"

CLASSIFICATION_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_classification"
CONFIDENCE_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_confidence"

# Zum Testen:
# None = alle Tweets klassifizieren
# z.B. 100 = nur erste 100 Tweets klassifizieren
TEST_LIMIT = None

SAVE_EVERY_N_ROWS = 10

client = OpenAI(timeout=30.0, max_retries=2)


# =========================
# Prompt
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
# Sicher speichern
# =========================

def save_checkpoint(df: pd.DataFrame):
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp.csv")

    df.to_csv(
        temp_file,
        index=False,
        encoding="utf-8-sig"
    )

    temp_file.replace(CHECKPOINT_FILE)


# =========================
# Hauptprogramm
# =========================

def main():
    print("Skript gestartet.")
    print(f"Modell: {MODEL}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print(f"Klassifikationsspalte: {CLASSIFICATION_COLUMN}")
    print(f"Konfidenzspalte: {CONFIDENCE_COLUMN}")

    if CHECKPOINT_FILE.exists():
        print(f"Checkpoint gefunden. Lade: {CHECKPOINT_FILE}")
        df = pd.read_csv(CHECKPOINT_FILE, low_memory=False)
    else:
        print(f"Kein Checkpoint gefunden. Lade Originaldatei: {INPUT_FILE}")

        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {INPUT_FILE}")

        df = pd.read_csv(INPUT_FILE, low_memory=False)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Spalte fehlt: {TEXT_COLUMN}")

    for col in [CLASSIFICATION_COLUMN, CONFIDENCE_COLUMN]:
        if col not in df.columns:
            df[col] = pd.NA

    if TEST_LIMIT is not None:
        df_to_process = df.head(TEST_LIMIT)
        print(f"Testmodus aktiv: Es werden maximal {TEST_LIMIT} Tweets verarbeitet.")
    else:
        df_to_process = df
        print("Testmodus deaktiviert: Es werden alle Tweets verarbeitet.")

    remaining = df_to_process[
        df_to_process[CLASSIFICATION_COLUMN].isna() |
        df_to_process[CONFIDENCE_COLUMN].isna()
    ]

    print(f"Bereits klassifiziert: {len(df_to_process) - len(remaining)}")
    print(f"Noch offen: {len(remaining)}")

    for counter, (idx, row) in enumerate(
        tqdm(
            remaining.iterrows(),
            total=len(remaining),
            desc=f"Tweets klassifizieren mit {MODEL}"
        ),
        start=1
    ):
        tweet_text = str(row[TEXT_COLUMN])

        result = classify_tweet(tweet_text)

        df.at[idx, CLASSIFICATION_COLUMN] = result["classification"]
        df.at[idx, CONFIDENCE_COLUMN] = result["confidence"]

        if counter % SAVE_EVERY_N_ROWS == 0:
            save_checkpoint(df)

    save_checkpoint(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFertig.")
    print(f"Ergebnis gespeichert unter: {OUTPUT_FILE}")
    print(f"Checkpoint gespeichert unter: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()