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
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_5 Final samples\Sample3"
)

INPUT_FILE = INPUT_DIR / "user_sample_400_users_per_state_nchs_proportional_with_existing_classifications.csv"

MODEL = "gpt-5.4-mini"
MODEL_NAME_FOR_COLUMNS = MODEL.replace("-", "_").replace(".", "_")

OUTPUT_FILE = INPUT_DIR / f"user_sample_400_users_per_state_nchs_llm_classified_{MODEL_NAME_FOR_COLUMNS}.csv"
CHECKPOINT_FILE = INPUT_DIR / f"user_sample_400_users_per_state_nchs_llm_checkpoint_{MODEL_NAME_FOR_COLUMNS}.csv"

USER_ID_COL = "user_id"
STATE_COL = "user_state"
STATE_NAME_COL = "user_state_name"
DATE_COL = "tweet_date"
TEXT_COL = "tweet_text"
TWEET_ID_COL = "tweet_id"

CLASSIFICATION_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_user_classification"
CONFIDENCE_COLUMN = f"{MODEL_NAME_FOR_COLUMNS}_user_confidence"

# Zum Testen:
# None = alle User klassifizieren
# z.B. 100 = nur erste 100 User klassifizieren
TEST_USER_LIMIT = None

SAVE_EVERY_N_USERS = 10

client = OpenAI(timeout=60.0, max_retries=2)


# =========================
# Prompt
# =========================

def build_prompt(tweet_texts: list[str]) -> str:
    tweets_formatted = "\n".join(
        [f"Tweet {i + 1}: {text}" for i, text in enumerate(tweet_texts)]
    )

    return f"""You are an expert political analyst.

Your task is to infer the overall political alignment of a single user based on their tweets contained within the <tweet> tags.

The tweets are ordered chronologically. The first tweet is the oldest, and the last tweet is the most recent. Infer the user's political alignment from the complete set of tweets, considering all tweets together as evidence.

Categories of political alignment:
1 = Republican-aligned (support for Republicans or opposition to Democrats)
-1 = Democrat-aligned (support for Democrats or opposition to Republicans)
0 = Insufficient evidence, mixed, balanced, unclear, conflicting political signals, or non-political.

Confidence:
Return a confidence score between 0.0 (completely uncertain) and 1.0 (completely certain). The confidence should reflect your certainty in the user's overall political alignment based on all tweets.

Output requirements:
- Return exactly one political alignment for the user.
- Do not classify or output classifications for individual tweets.
- Return ONLY a valid JSON object.
- Do not include explanations, markdown, or additional text.
- The JSON object must contain exactly these two keys:
  - "classification"
  - "confidence"

Desired JSON structure:
{{"classification": <integer>, "confidence": <float>}}

<tweet>
{tweets_formatted}
</tweet>"""


# =========================
# API-Klassifikation
# =========================

def classify_user(tweet_texts: list[str], max_retries: int = 3) -> dict:
    prompt = build_prompt(tweet_texts)

    DEBUG_PROMPT = False

    if DEBUG_PROMPT:
        debug_file = INPUT_DIR / "debug_prompt.txt"

        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(prompt)

        print(f"Prompt gespeichert unter: {debug_file}")

        

    json_schema = {
        "name": "user_political_alignment_schema",
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

    required_cols = [
        USER_ID_COL,
        STATE_COL,
        STATE_NAME_COL,
        DATE_COL,
        TEXT_COL,
        TWEET_ID_COL
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Fehlende Spalten: {missing_cols}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    for col in [CLASSIFICATION_COLUMN, CONFIDENCE_COLUMN]:
        if col not in df.columns:
            df[col] = pd.NA

    user_groups = (
        df[[STATE_COL, STATE_NAME_COL, USER_ID_COL]]
        .drop_duplicates()
        .sort_values([STATE_COL, USER_ID_COL])
        .reset_index(drop=True)
    )

    if TEST_USER_LIMIT is not None:
        user_groups = user_groups.head(TEST_USER_LIMIT)
        print(f"Testmodus aktiv: Es werden maximal {TEST_USER_LIMIT} User verarbeitet.")
    else:
        print("Testmodus deaktiviert: Es werden alle User verarbeitet.")

    already_classified = (
        df.groupby([STATE_COL, USER_ID_COL])[[CLASSIFICATION_COLUMN, CONFIDENCE_COLUMN]]
        .apply(lambda x: x[CLASSIFICATION_COLUMN].notna().all() and x[CONFIDENCE_COLUMN].notna().all())
        .reset_index(name="is_classified")
    )

    user_groups = user_groups.merge(
        already_classified,
        on=[STATE_COL, USER_ID_COL],
        how="left"
    )

    user_groups["is_classified"] = user_groups["is_classified"].fillna(False)

    remaining_users = user_groups[user_groups["is_classified"] == False]

    print(f"Bereits klassifizierte User im aktuellen Laufbereich: {len(user_groups) - len(remaining_users)}")
    print(f"Noch offene User im aktuellen Laufbereich: {len(remaining_users)}")

    for counter, (_, user_row) in enumerate(
        tqdm(
            remaining_users.iterrows(),
            total=len(remaining_users),
            desc=f"User klassifizieren mit {MODEL}"
        ),
        start=1
    ):
        state = user_row[STATE_COL]
        user_id = user_row[USER_ID_COL]

        user_tweets_df = (
            df[
                (df[STATE_COL] == state) &
                (df[USER_ID_COL] == user_id)
            ]
            .sort_values(DATE_COL)
        )

        tweet_texts = (
            user_tweets_df[TEXT_COL]
            .dropna()
            .astype(str)
            .str.strip()
        )

        tweet_texts = [text for text in tweet_texts if text]

        if not tweet_texts:
            result = {
                "classification": None,
                "confidence": None
            }
        else:
            result = classify_user(tweet_texts)

        mask = (
            (df[STATE_COL] == state) &
            (df[USER_ID_COL] == user_id)
        )

        df.loc[mask, CLASSIFICATION_COLUMN] = result["classification"]
        df.loc[mask, CONFIDENCE_COLUMN] = result["confidence"]

        if counter % SAVE_EVERY_N_USERS == 0:
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