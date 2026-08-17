from pathlib import Path
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


INPUT_DIR = Path(r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\07_llm_classification")
INPUT_FILE = INPUT_DIR / "manuel_labeling_500_tweets.xlsx"

MODEL = "gpt-5.4"
MODEL_NAME_FOR_COLUMNS = MODEL.replace("-", "_").replace(".", "_")

OUTPUT_FILE = INPUT_DIR / "manuel_labeling_500_tweets_llm_classified_v4.3.xlsx"
CHECKPOINT_FILE = INPUT_DIR / "manuel_labeling_500_tweets_llm_checkpoint.xlsx"

TEXT_COLUMN = "tweet_text_for_labeling"

CLASSIFICATION_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_classification"
CONFIDENCE_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_confidence"

client = OpenAI(timeout=30.0, max_retries=2)


def build_prompt(tweet_text: str) -> str:
    return f"""You are an expert political analyst. Classify the political alignment expressed or implied by the tweet inside the <tweet> tags within the context of US politics.
Categories:
1 = Republican-aligned (support for Republicans or opposition to Democrats)
-1 = Democrat-aligned (support for Democrats or opposition to Republicans)
0 = Neutral, balanced, unclear, or non-political

Confidence score:
A float between 0.0 (completely uncertain) and 1.0 (absolutely certain).

Output Format:
Return ONLY a valid JSON object. Do not include markdown formatting, explanations, or additional text.

Desired JSON structure:
{{"classification": 0, "confidence": 0.5}}

<tweet>
{tweet_text}
</tweet>"""


def classify_tweet(tweet_text: str, max_retries: int = 3) -> dict:
    prompt = build_prompt(tweet_text)

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=0,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "tweet_classification",
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
                        },
                        "strict": True
                    }
                }
            )

            result = json.loads(response.output_text)

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


def save_checkpoint(df: pd.DataFrame):
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp.xlsx")
    df.to_excel(temp_file, index=False)
    temp_file.replace(CHECKPOINT_FILE)


def main():
    print("Skript gestartet...")
    print(f"Aktuelles Modell: {MODEL}")
    print(f"Klassifikationsspalte: {CLASSIFICATION_COLUMN}")
    print(f"Konfidenzspalte: {CONFIDENCE_COLUMN}")

    if CHECKPOINT_FILE.exists():
        print(f"Checkpoint gefunden. Lade: {CHECKPOINT_FILE}")
        df = pd.read_excel(CHECKPOINT_FILE)
    else:
        print(f"Kein Checkpoint gefunden. Lade Originaldatei: {INPUT_FILE}")
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {INPUT_FILE}")
        df = pd.read_excel(INPUT_FILE)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Spalte fehlt: {TEXT_COLUMN}")

    if CLASSIFICATION_COLUMN not in df.columns:
        df[CLASSIFICATION_COLUMN] = None

    if CONFIDENCE_COLUMN not in df.columns:
        df[CONFIDENCE_COLUMN] = None

    TEST_LIMIT = 100

    df_test = df.head(TEST_LIMIT)

    remaining = df_test[
        df_test[CLASSIFICATION_COLUMN].isna() | df_test[CONFIDENCE_COLUMN].isna()
    ]

    print(f"Bereits mit {MODEL} klassifiziert: {len(df) - len(remaining)}")
    print(f"Noch offen für {MODEL}: {len(remaining)}")

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

        if counter % 10 == 0:
            save_checkpoint(df)

    save_checkpoint(df)
    
    

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Fertig. Ergebnis gespeichert unter: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()