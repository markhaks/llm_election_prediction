"""
Dieses Skript erstellt eine proportionale User-Level-Stichprobe.

Ablauf:
- alle CSV-Dateien aus dem Input-Ordner laden
- Zeitraum auf 10.10.2024 bis 04.11.2024 beschränken
- Tweets pro Nutzer zusammenhalten
- Nutzer anhand von nchs_2023_category in zwei Gruppen einteilen:
  Kategorie 1-3 und Kategorie 4-6
- pro Bundesstaat die Zielverteilung aus einer Excel-Datei laden
- pro Bundesstaat bis zu 400 Nutzer proportional zur Bevölkerung ziehen
- wenn nicht genug Nutzer verfügbar sind, wird die größtmögliche Stichprobe gezogen,
  bei der das Verhältnis möglichst erhalten bleibt
- alle Tweets der ausgewählten Nutzer innerhalb des Zeitraums behalten
- Tweets pro Nutzer chronologisch sortieren

Output:
- user_sample_400_users_per_state_nchs_proportional_tweets.csv
- user_sample_nchs_proportional_summary.csv
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

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_5 Final samples\Sample3"
)

QUOTA_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_5 Final samples\Sample3\Aufteilung Staat in Kategorien.xlsx"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATTERN = "*.csv"

USER_ID_COL = "user_id"
STATE_COL = "user_state"
STATE_NAME_COL = "user_state_name"
DATE_COL = "tweet_date"
TEXT_COL = "tweet_text"
TWEET_ID_COL = "tweet_id"
NCHS_COL = "nchs_2023_category"

START_DATE = "2024-10-10"
END_DATE = "2024-11-04"

USERS_PER_STATE = 400
RANDOM_SEED = 42


# =========================
# Hilfsfunktion: proportionale Quoten berechnen
# =========================

def largest_remainder_allocation(total_n, proportions):
    """
    Verteilt total_n ganzzahlig nach proportions.
    Beispiel: total_n=400, proportions={1-3:0.8, 4-6:0.2}
    Ergebnis: {1-3:320, 4-6:80}
    """
    raw = {k: total_n * v for k, v in proportions.items()}
    floors = {k: int(np.floor(v)) for k, v in raw.items()}

    remaining = total_n - sum(floors.values())

    remainders = sorted(
        raw.keys(),
        key=lambda k: raw[k] - floors[k],
        reverse=True
    )

    allocation = floors.copy()

    for k in remainders[:remaining]:
        allocation[k] += 1

    return allocation


def find_feasible_allocation(target_total, proportions, available):
    """
    Sucht die größte mögliche Stichprobengröße <= target_total,
    bei der die proportionale Aufteilung mit den verfügbaren Nutzern möglich ist.
    """
    for total_n in range(target_total, 0, -1):
        allocation = largest_remainder_allocation(total_n, proportions)

        feasible = True

        for group, n_required in allocation.items():
            if n_required > available.get(group, 0):
                feasible = False
                break

        if feasible:
            return total_n, allocation

    return 0, {group: 0 for group in proportions.keys()}


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
    TEXT_COL,
    NCHS_COL
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Fehlende Spalten in CSV-Dateien: {missing_cols}")


# =========================
# Quoten-Datei laden
# =========================

quota_df = pd.read_excel(QUOTA_FILE)

required_quota_cols = [
    "Bundesstaat",
    "Kategorie",
    "Verteilung im Staat"
]

missing_quota_cols = [col for col in required_quota_cols if col not in quota_df.columns]

if missing_quota_cols:
    raise ValueError(f"Fehlende Spalten in Quoten-Datei: {missing_quota_cols}")

quota_df["Bundesstaat"] = quota_df["Bundesstaat"].astype(str).str.strip()
quota_df["Kategorie"] = quota_df["Kategorie"].astype(str).str.strip()
quota_df["Verteilung im Staat"] = pd.to_numeric(
    quota_df["Verteilung im Staat"],
    errors="coerce"
)

quota_pivot = (
    quota_df
    .pivot_table(
        index="Bundesstaat",
        columns="Kategorie",
        values="Verteilung im Staat",
        aggfunc="sum"
    )
    .reset_index()
)

quota_pivot = quota_pivot.rename(columns={
    "Kategorie 1-3": "target_share_1_3",
    "Kategorie 4-6": "target_share_4_6"
})

for col in ["target_share_1_3", "target_share_4_6"]:
    if col not in quota_pivot.columns:
        quota_pivot[col] = 0.0

quota_pivot[["target_share_1_3", "target_share_4_6"]] = quota_pivot[
    ["target_share_1_3", "target_share_4_6"]
].fillna(0.0)


# =========================
# Bereinigung
# =========================

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# Aus Werten wie "1 - Large central metro" nur die Zahl am Anfang extrahieren
df[NCHS_COL] = (
    df[NCHS_COL]
    .astype(str)
    .str.extract(r"^(\d+)")[0]
)

df[NCHS_COL] = pd.to_numeric(df[NCHS_COL], errors="coerce")

df = df.dropna(subset=[
    TWEET_ID_COL,
    USER_ID_COL,
    STATE_COL,
    STATE_NAME_COL,
    DATE_COL,
    TEXT_COL,
    NCHS_COL
])

df[TEXT_COL] = df[TEXT_COL].astype(str)
df = df[df[TEXT_COL].str.strip() != ""]

df = df[df[NCHS_COL].isin([1, 2, 3, 4, 5, 6])].copy()

df["nchs_group"] = np.where(
    df[NCHS_COL].isin([1, 2, 3]),
    "Kategorie 1-3",
    "Kategorie 4-6"
)

start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df = df[
    (df[DATE_COL] >= start_dt) &
    (df[DATE_COL] <= end_dt)
].copy()

df = df.drop_duplicates(subset=[TWEET_ID_COL])

df["date_only"] = df[DATE_COL].dt.date

print(f"\nTweets nach Filterung: {len(df):,}")
print(f"Bundesstaaten in Tweetdaten: {df[STATE_COL].nunique()}")
print(f"Eindeutige Nutzer gesamt: {df[USER_ID_COL].nunique():,}")


# =========================
# User-Level-Datensatz erstellen
# =========================

users_df = (
    df[
        [
            USER_ID_COL,
            STATE_COL,
            STATE_NAME_COL,
            "nchs_group"
        ]
    ]
    .drop_duplicates(subset=[USER_ID_COL, STATE_COL])
    .copy()
)


# =========================
# Proportionale User-Stichprobe ziehen
# =========================

selected_user_rows = []
summary_rows = []

for state, state_users in users_df.groupby(STATE_COL):
    state = str(state).strip()

    state_quota = quota_pivot[quota_pivot["Bundesstaat"] == state]

    if state_quota.empty:
        print(f"Warnung: Keine Quoten für Bundesstaat {state}. Bundesstaat wird übersprungen.")
        continue

    target_share_1_3 = float(state_quota["target_share_1_3"].iloc[0])
    target_share_4_6 = float(state_quota["target_share_4_6"].iloc[0])

    share_sum = target_share_1_3 + target_share_4_6

    if share_sum <= 0:
        print(f"Warnung: Quoten für {state} ergeben 0. Bundesstaat wird übersprungen.")
        continue

    proportions = {
        "Kategorie 1-3": target_share_1_3 / share_sum,
        "Kategorie 4-6": target_share_4_6 / share_sum
    }

    available = (
        state_users
        .groupby("nchs_group")[USER_ID_COL]
        .nunique()
        .to_dict()
    )

    total_sampled, allocation = find_feasible_allocation(
        target_total=USERS_PER_STATE,
        proportions=proportions,
        available=available
    )

    if total_sampled == 0:
        print(f"Warnung: Für {state} konnten keine Nutzer gezogen werden.")
        continue

    for group, n_to_sample in allocation.items():
        group_users = state_users[state_users["nchs_group"] == group].copy()

        available_users = group_users[USER_ID_COL].nunique()

        if n_to_sample > 0:
            sampled_group_users = group_users.sample(
                n=n_to_sample,
                random_state=RANDOM_SEED
            ).copy()

            sampled_group_users["target_share"] = proportions[group]
            sampled_group_users["sampled_users_in_group"] = n_to_sample
            sampled_group_users["available_users_in_group"] = available_users

            selected_user_rows.append(sampled_group_users)

        summary_rows.append({
            STATE_COL: state,
            STATE_NAME_COL: state_users[STATE_NAME_COL].iloc[0],
            "nchs_group": group,
            "target_share": proportions[group],
            "target_users_if_400": largest_remainder_allocation(
                USERS_PER_STATE,
                proportions
            )[group],
            "available_users": available_users,
            "sampled_users": n_to_sample,
            "actual_share_in_sample": n_to_sample / total_sampled if total_sampled > 0 else 0,
            "total_sampled_users_in_state": total_sampled
        })

selected_users = pd.concat(selected_user_rows, ignore_index=True)

selected_users = selected_users.sort_values(
    [STATE_COL, "nchs_group", USER_ID_COL]
).reset_index(drop=True)

selected_users["user_sample_rank_within_state"] = (
    selected_users
    .groupby(STATE_COL)
    .cumcount() + 1
)


# =========================
# Alle Tweets der ausgewählten Nutzer behalten
# =========================

sample = df.merge(
    selected_users[
        [
            USER_ID_COL,
            STATE_COL,
            "nchs_group",
            "user_sample_rank_within_state"
        ]
    ],
    on=[USER_ID_COL, STATE_COL, "nchs_group"],
    how="inner"
)

sample["tweets_of_user"] = (
    sample
    .groupby([STATE_COL, USER_ID_COL])[TWEET_ID_COL]
    .transform("count")
)

sample = sample.sort_values(
    [
        STATE_COL,
        "user_sample_rank_within_state",
        DATE_COL
    ]
).reset_index(drop=True)

sample["tweet_rank_within_user"] = (
    sample
    .groupby([STATE_COL, USER_ID_COL])
    .cumcount() + 1
)


# =========================
# Summary erstellen
# =========================

summary = pd.DataFrame(summary_rows)

tweet_summary = (
    sample
    .groupby([STATE_COL, "nchs_group"])
    .agg(
        tweets=(TWEET_ID_COL, "count"),
        unique_users=(USER_ID_COL, "nunique"),
        first_date=(DATE_COL, "min"),
        last_date=(DATE_COL, "max"),
        days_covered=("date_only", "nunique")
    )
    .reset_index()
)

summary = summary.merge(
    tweet_summary,
    on=[STATE_COL, "nchs_group"],
    how="left"
)

summary["tweets"] = summary["tweets"].fillna(0).astype(int)
summary["unique_users"] = summary["unique_users"].fillna(0).astype(int)
summary["difference_target_actual_share"] = (
    summary["actual_share_in_sample"] - summary["target_share"]
)

summary = summary.sort_values(
    [STATE_COL, "nchs_group"]
).reset_index(drop=True)


# =========================
# Dateien speichern
# =========================

output_file = OUTPUT_DIR / "user_sample_400_users_per_state_nchs_proportional_tweets.csv"

sample.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

summary_file = OUTPUT_DIR / "user_sample_nchs_proportional_summary.csv"

summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nFertig.")
print(f"Proportionale User-Level-Stichprobe: {output_file}")
print(f"Summary: {summary_file}")

print("\nGesamtübersicht:")
print(
    summary
    .groupby(STATE_COL)
    .agg(
        sampled_users=("sampled_users", "sum"),
        tweets=("tweets", "sum")
    )
    .reset_index()
)