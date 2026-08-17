"""
Dieses Skript erstellt eine User-Level-Stichprobe.

Ablauf:
- alle CSV-Dateien aus dem Input-Ordner laden
- Zeitraum auf 10.10.2024 bis 04.11.2024 beschränken
- Tweets pro Nutzer zusammenhalten
- pro Bundesstaat 400 Nutzer zufällig auswählen
- alle Tweets dieser Nutzer innerhalb des Zeitraums behalten
- Tweets pro Nutzer chronologisch sortieren

Output:
- user_sample_400_users_per_state_tweets.csv
- user_sample_summary.csv
"""

from pathlib import Path
import pandas as pd

# =========================
# Einstellungen
# =========================

INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2c new data"
)

OUTPUT_DIR = Path(r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_5 Final samples\sampe2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATTERN = "*.csv"

USER_ID_COL = "user_id"
STATE_COL = "user_state"
STATE_NAME_COL = "user_state_name"
DATE_COL = "tweet_date"
TEXT_COL = "tweet_text"
TWEET_ID_COL = "tweet_id"

START_DATE = "2024-10-10"
END_DATE = "2024-11-04"

USERS_PER_STATE = 400
RANDOM_SEED = 42


# =========================
# Daten laden
# =========================

csv_files = list(INPUT_DIR.glob(CSV_PATTERN))

if not csv_files:
    raise FileNotFoundError(f"Keine CSV-Dateien gefunden in: {INPUT_DIR}")

dfs = []

for file in csv_files:
    print(f"Lade: {file.name}")
    temp = pd.read_csv(file, low_memory=False)
    temp["source_file"] = file.name
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

required_cols = [
    TWEET_ID_COL,
    USER_ID_COL,
    STATE_COL,
    STATE_NAME_COL,
    DATE_COL,
    TEXT_COL
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Fehlende Spalten: {missing_cols}")


# =========================
# Bereinigung
# =========================

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

df = df.dropna(subset=[
    TWEET_ID_COL,
    USER_ID_COL,
    STATE_COL,
    STATE_NAME_COL,
    DATE_COL,
    TEXT_COL
])

df[TEXT_COL] = df[TEXT_COL].astype(str)
df = df[df[TEXT_COL].str.strip() != ""]

start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df = df[
    (df[DATE_COL] >= start_dt) &
    (df[DATE_COL] <= end_dt)
].copy()

df = df.drop_duplicates(subset=[TWEET_ID_COL])

df["date_only"] = df[DATE_COL].dt.date

print(f"\nTweets nach Filterung: {len(df):,}")
print(f"Bundesstaaten: {df[STATE_COL].nunique()}")
print(f"Eindeutige Nutzer gesamt: {df[USER_ID_COL].nunique():,}")


# =========================
# User-Level-Stichprobe ziehen
# =========================

selected_user_rows = []

for state, state_df in df.groupby(STATE_COL):
    users = (
        state_df[[USER_ID_COL, STATE_COL, STATE_NAME_COL]]
        .drop_duplicates(subset=[USER_ID_COL])
        .copy()
    )

    available_users = len(users)
    n_users = min(USERS_PER_STATE, available_users)

    if available_users < USERS_PER_STATE:
        print(
            f"Warnung: {state} hat nur {available_users} eindeutige Nutzer. "
            f"Statt {USERS_PER_STATE} werden {available_users} gezogen."
        )

    sampled_users = users.sample(
        n=n_users,
        random_state=RANDOM_SEED
    )

    sampled_users["user_sample_rank_within_state"] = range(1, n_users + 1)

    selected_user_rows.append(sampled_users)

selected_users = pd.concat(selected_user_rows, ignore_index=True)


# =========================
# Alle Tweets der ausgewählten Nutzer behalten
# =========================

sample = df.merge(
    selected_users[
        [USER_ID_COL, STATE_COL, "user_sample_rank_within_state"]
    ],
    on=[USER_ID_COL, STATE_COL],
    how="inner"
)

sample["tweets_of_user"] = (
    sample
    .groupby([STATE_COL, USER_ID_COL])[TWEET_ID_COL]
    .transform("count")
)

sample = sample.sort_values(
    [STATE_COL, "user_sample_rank_within_state", DATE_COL]
).reset_index(drop=True)

sample["tweet_rank_within_user"] = (
    sample
    .groupby([STATE_COL, USER_ID_COL])
    .cumcount() + 1
)


# =========================
# Dateien speichern
# =========================

output_file = OUTPUT_DIR / "user_sample_400_users_per_state_tweets.csv"

sample.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

summary = (
    sample
    .groupby([STATE_COL, STATE_NAME_COL])
    .agg(
        sampled_users=(USER_ID_COL, "nunique"),
        tweets=(TWEET_ID_COL, "count"),
        avg_tweets_per_user=(TWEET_ID_COL, lambda x: len(x) / sample.loc[x.index, USER_ID_COL].nunique()),
        first_date=(DATE_COL, "min"),
        last_date=(DATE_COL, "max"),
        days_covered=("date_only", "nunique")
    )
    .reset_index()
)

summary_file = OUTPUT_DIR / "user_sample_summary.csv"

summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nFertig.")
print(f"User-Level-Stichprobe: {output_file}")
print(f"Übersicht: {summary_file}")