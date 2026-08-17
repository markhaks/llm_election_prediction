"""
Dieses Skript erstellt vier verschachtelte Tweet-Stichproben pro US-Bundesstaat.

Kriterien:
- Zeitraum: 10.10.2024 bis 04.11.2024
- pro Stichprobe keine doppelten Nutzer
- Stichprobengrößen: 100, 200, 300 und 400 Tweets je Bundesstaat
- kleinere Stichproben sind echte Teilmengen der größeren Stichproben:
  100 ⊂ 200 ⊂ 300 ⊂ 400
- die Tweets werden pro Bundesstaat über die Tage verteilt
- die Tagesverteilung erfolgt proportional zur verfügbaren Tweet-Anzahl je Tag
- verfügbare Tage werden möglichst berücksichtigt, sofern genug eindeutige Nutzer vorhanden sind

Methodisch handelt es sich nicht um eine einfache Zufallsstichprobe,
sondern um eine nach Bundesstaat und Tag eingeschränkte/geschichtete Zufallsstichprobe.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# Einstellungen
# =========================

INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2c new data"
)

OUTPUT_DIR = INPUT_DIR / "samples_nested_proportional_days"
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

RANDOM_SEED = 42
SAMPLE_SIZES = [100, 200, 300, 400]


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

df["date_only"] = df[DATE_COL].dt.date

df = df.drop_duplicates(subset=[TWEET_ID_COL])

print(f"\nTweets nach Filterung: {len(df):,}")
print(f"Bundesstaaten: {df[STATE_COL].nunique()}")


# =========================
# Proportional-tagesbalanciertes Sampling
# =========================

def allocate_proportional_counts(day_counts: pd.Series, target_n: int) -> dict:
    """
    Verteilt target_n proportional auf Tage.
    Jeder verfügbare Tag erhält möglichst mindestens 1 Tweet,
    sofern target_n >= Anzahl verfügbarer Tage.
    """
    day_counts = day_counts.sort_index()
    dates = list(day_counts.index)
    total_available = int(day_counts.sum())

    if target_n <= 0 or total_available == 0:
        return {date: 0 for date in dates}

    if target_n >= len(dates):
        allocation = {date: 1 for date in dates}
        remaining = target_n - len(dates)
    else:
        allocation = {date: 0 for date in dates}
        remaining = target_n

    if remaining <= 0:
        return allocation

    available_after_min = {
        date: max(int(day_counts.loc[date]) - allocation[date], 0)
        for date in dates
    }

    total_after_min = sum(available_after_min.values())

    if total_after_min == 0:
        return allocation

    raw = {
        date: remaining * available_after_min[date] / total_after_min
        for date in dates
    }

    floor_alloc = {
        date: int(np.floor(raw[date]))
        for date in dates
    }

    for date in dates:
        allocation[date] += floor_alloc[date]

    assigned = sum(floor_alloc.values())
    still_missing = remaining - assigned

    remainders = sorted(
        dates,
        key=lambda date: raw[date] - floor_alloc[date],
        reverse=True
    )

    for date in remainders:
        if still_missing <= 0:
            break

        if allocation[date] < int(day_counts.loc[date]):
            allocation[date] += 1
            still_missing -= 1

    return allocation


def create_nested_proportional_sample(
    group: pd.DataFrame,
    target_n: int,
    seed: int
) -> pd.DataFrame:
    state = group[STATE_COL].iloc[0]

    # Pro Nutzer nur ein Tweet: zufällig einen Tweet je User auswählen
    unique_user_pool = (
        group
        .sample(frac=1, random_state=seed)
        .drop_duplicates(subset=[USER_ID_COL], keep="first")
        .copy()
    )

    available_n = len(unique_user_pool)

    if available_n < target_n:
        print(
            f"Warnung: {state} hat nur {available_n} eindeutige Nutzer. "
            f"Statt {target_n} werden {available_n} gezogen."
        )
        target_n = available_n

    if target_n == 0:
        return pd.DataFrame(columns=group.columns)

    day_counts = unique_user_pool["date_only"].value_counts().sort_index()
    allocation = allocate_proportional_counts(day_counts, target_n)

    sampled_parts = []

    for date, n_for_day in allocation.items():
        if n_for_day <= 0:
            continue

        day_group = unique_user_pool[unique_user_pool["date_only"] == date]

        sampled_day = day_group.sample(
            n=min(n_for_day, len(day_group)),
            random_state=seed
        )

        sampled_parts.append(sampled_day)

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # Falls durch Rundung oder knappe Tagespools noch etwas fehlt, aus Rest auffüllen
    missing = target_n - len(sampled)

    if missing > 0:
        remaining_pool = unique_user_pool[
            ~unique_user_pool[USER_ID_COL].isin(sampled[USER_ID_COL])
        ]

        fill_n = min(missing, len(remaining_pool))

        if fill_n > 0:
            fill_sample = remaining_pool.sample(
                n=fill_n,
                random_state=seed
            )

            sampled = pd.concat([sampled, fill_sample], ignore_index=True)

    # Wichtig:
    # Ranking wird zufällig, aber innerhalb der proportionalen Tagesstruktur vergeben.
    # Dadurch bleiben 100, 200 und 300 echte Teilmengen der 400er-Stichprobe.
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled["sample_rank_within_state"] = range(1, len(sampled) + 1)

    return sampled


# =========================
# 400er Gesamtstichprobe erstellen
# =========================

max_sample_size = max(SAMPLE_SIZES)

all_state_samples = []

for state, group in df.groupby(STATE_COL):
    state_sample = create_nested_proportional_sample(
        group=group,
        target_n=max_sample_size,
        seed=RANDOM_SEED
    )

    all_state_samples.append(state_sample)

sample_400 = pd.concat(all_state_samples, ignore_index=True)

sample_400 = sample_400.sort_values(
    [STATE_COL, "sample_rank_within_state"]
).reset_index(drop=True)


# =========================
# Dateien speichern
# =========================

for size in SAMPLE_SIZES:
    sample_size_df = sample_400[
        sample_400["sample_rank_within_state"] <= size
    ].copy()

    output_file = OUTPUT_DIR / f"sample_{size}_tweets_per_state.csv"

    sample_size_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Gespeichert: {output_file}")


# =========================
# Kontrollübersicht speichern
# =========================

summary_rows = []

for size in SAMPLE_SIZES:
    temp = sample_400[
        sample_400["sample_rank_within_state"] <= size
    ].copy()

    summary = (
        temp
        .groupby([STATE_COL, STATE_NAME_COL])
        .agg(
            tweets=(TWEET_ID_COL, "count"),
            unique_users=(USER_ID_COL, "nunique"),
            first_date=(DATE_COL, "min"),
            last_date=(DATE_COL, "max"),
            days_covered=("date_only", "nunique")
        )
        .reset_index()
    )

    summary["target_tweets_per_state"] = size
    summary_rows.append(summary)

summary_all = pd.concat(summary_rows, ignore_index=True)

summary_file = OUTPUT_DIR / "sample_summary.csv"

summary_all.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


# =========================
# Tagesverteilung speichern
# =========================

daily_rows = []

for size in SAMPLE_SIZES:
    temp = sample_400[
        sample_400["sample_rank_within_state"] <= size
    ].copy()

    daily = (
        temp
        .groupby([STATE_COL, STATE_NAME_COL, "date_only"])
        .agg(
            tweets=(TWEET_ID_COL, "count"),
            unique_users=(USER_ID_COL, "nunique")
        )
        .reset_index()
    )

    daily["target_tweets_per_state"] = size
    daily_rows.append(daily)

daily_summary = pd.concat(daily_rows, ignore_index=True)

daily_summary_file = OUTPUT_DIR / "sample_daily_distribution.csv"

daily_summary.to_csv(
    daily_summary_file,
    index=False,
    encoding="utf-8-sig"
)


print("\nFertig.")
print(f"Output-Ordner: {OUTPUT_DIR}")
print(f"Kontrollübersicht: {summary_file}")
print(f"Tagesverteilung: {daily_summary_file}")