from pathlib import Path
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Pfade definieren
INPUT_DIR = Path(r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\07_llm_classification")
INPUT_FILE = INPUT_DIR / "manuel_labeling_500_tweets.xlsx"

MODEL = "gpt-5.4"  # Hinweis: Setze hier dein exaktes Modell ein (z. B. gpt-4o oder gpt-4o-mini)
MODEL_NAME_FOR_COLUMNS = MODEL.replace("-", "_").replace(".", "_")

OUTPUT_FILE = INPUT_DIR / "manuel_labeling_500_tweets_llm_classified_v3.xlsx"
CHECKPOINT_FILE = INPUT_DIR / "manuel_labeling_500_tweets_llm_checkpoint.xlsx"

TEXT_COLUMN = "tweet_text_for_labeling"

# Spaltennamen für das DataFrame
REASONING_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_reasoning"
CLASSIFICATION_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_classification"
CONFIDENCE_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_confidence"

client = OpenAI(timeout=30.0, max_retries=2)


def build_prompt(tweet_text: str) -> str:
    """Erstellt den Prompt mit klaren Definitionen und Few-Shot-Beispielen."""
    return f"""You are an expert political analyst. Carefully analyze the political keywords, figures, and underlying bias in the tweet inside the <tweet> tags.

Before choosing the final category, mentally evaluate if the text contains conservative/Republican or progressive/Democratic talking points, or if it is purely factual/neutral.    

Categories:
1 = Republican-aligned (support for Republicans, conservative talking points, or opposition to Democrats)
-1 = Democrat-aligned (support for Democrats, progressive talking points, or opposition to Republicans)
0 = Neutral, balanced, factual reporting, unclear, or non-political

Guidelines:
- First, provide a short, objective reasoning identifying key political figures, themes, or biases.
- Then, select the final classification (-1, 0, or 1) and your confidence score.

Examples:
Tweet: "The GOP is weak."
Reasoning: Not clear.
Classification: 0

Tweet: "nearly a quarter of a million votes in play. How else can you try to elect a phony. JOE BIDEN won the primary, he garnered ALL the electoral votes to clinch the Democratic nomination. Kamala Harris didn't do A THING! Not one thing. In effect, they took Joe Biden's nomination"
Reasoning: Not clear, could be a democrat criticising his own party.
Classification: 0

<tweet>
{tweet_text}
</tweet>"""


def classify_tweet(tweet_text: str, max_retries: int = 3) -> dict:
    prompt = build_prompt(tweet_text)

    # WICHTIG: Das 'reasoning'-Feld steht an erster Stelle, damit das Modell 
    # zuerst nachdenkt (Chain of Thought), bevor es die Entscheidung fällen muss.
    json_schema = {
        "name": "tweet_classification_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "integer",
                    "enum": [-1, 0, 1],
                    "description": "The final political alignment score."
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the classification."
                }
            },
            "required": ["classification", "confidence"],
            "additionalProperties": False
        }
    }

    for attempt in range(max_retries):
        try:
            # Umstellung auf chat.completions für saubere Trennung von System- und User-Rolle
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,  # Deterministische Ergebnisse erzwingen
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                },
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert political analyst specializing in US politics. Analyze tweets objectively and systematically."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )

            # Extraktion der JSON-Antwort
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
        "reasoning": None,
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

    # Neue Spalten initialisieren, falls sie noch nicht existieren
    for col in [CLASSIFICATION_COLUMN, CONFIDENCE_COLUMN]:
        if col not in df.columns:
            df[col] = None

    TEST_LIMIT = 100
    df_test = df.head(TEST_LIMIT)

    # Zeilen finden, die noch nicht klassifiziert wurden
    remaining = df_test[
        df_test[CLASSIFICATION_COLUMN].isna() | 
        df_test[CONFIDENCE_COLUMN].isna()
    ]

    print(f"Bereits mit {MODEL} klassifiziert: {TEST_LIMIT - len(remaining)}")
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

        # Ergebnisse ins DataFrame schreiben
        df.at[idx, CLASSIFICATION_COLUMN] = result["classification"]
        df.at[idx, CONFIDENCE_COLUMN] = result["confidence"]

        if counter % 10 == 0:
            save_checkpoint(df)

    save_checkpoint(df)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Fertig. Ergebnis gespeichert unter: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()