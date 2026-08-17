"""
Dieses Skript übernimmt bereits vorhandene User-Level-LLM-Klassifizierungen
aus einer bereits klassifizierten Stichprobe in eine neue NCHS-proportionale Stichprobe.

Abgleich:
- ausschließlich über user_id

Übernommen werden:
- gpt_5_4_mini_user_classification
- gpt_5_4_mini_user_confidence

Output:
- user_sample_400_users_per_state_nchs_proportional_with_existing_classifications.csv
- classification_transfer_summary.csv
"""

from pathlib import Path
import pandas as pd

# =========================
# Einstellungen
# =========================

CLASSIFIED_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_5 Final samples\sampe2\user_sample_400_users_per_state_llm_classified_gpt_5_4_mini.csv"
)

NEW_SAMPLE_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_5 Final samples\Sample3\user_sample_400_users_per_state_nchs_proportional_tweets.csv"
)

OUTPUT_DIR = NEW_SAMPLE_FILE.parent

OUTPUT_FILE = OUTPUT_DIR / "user_sample_400_users_per_state_nchs_proportional_with_existing_classifications.csv"
SUMMARY_FILE = OUTPUT_DIR / "classification_transfer_summary.csv"

USER_ID_COL = "user_id"
STATE_COL = "user_state"

CLASSIFICATION_COL = "gpt_5_4_mini_user_classification"
CONFIDENCE_COL = "gpt_5_4_mini_user_confidence"


# =========================
# Dateien laden
# =========================

classified_df = pd.read_csv(CLASSIFIED_FILE, low_memory=False)
new_df = pd.read_csv(NEW_SAMPLE_FILE, low_memory=False)

required_classified_cols = [
    USER_ID_COL,
    CLASSIFICATION_COL,
    CONFIDENCE_COL
]

required_new_cols = [
    USER_ID_COL,
    STATE_COL
]

missing_classified = [
    col for col in required_classified_cols
    if col not in classified_df.columns
]

missing_new = [
    col for col in required_new_cols
    if col not in new_df.columns
]

if missing_classified:
    raise ValueError(f"Fehlende Spalten in klassifizierter Datei: {missing_classified}")

if missing_new:
    raise ValueError(f"Fehlende Spalten in neuer Datei: {missing_new}")


# =========================
# Klassifizierte User vorbereiten
# =========================

classified_users = (
    classified_df[
        [
            USER_ID_COL,
            CLASSIFICATION_COL,
            CONFIDENCE_COL
        ]
    ]
    .dropna(subset=[CLASSIFICATION_COL], how="all")
    .drop_duplicates(subset=[USER_ID_COL], keep="first")
    .copy()
)

print(f"Klassifizierte User in alter Datei: {len(classified_users):,}")


# =========================
# Falls Spalten schon in neuer Datei existieren: entfernen
# =========================

cols_to_drop = [
    col for col in [CLASSIFICATION_COL, CONFIDENCE_COL]
    if col in new_df.columns
]

if cols_to_drop:
    new_df = new_df.drop(columns=cols_to_drop)


# =========================
# Klassifizierungen übertragen
# =========================

merged_df = new_df.merge(
    classified_users,
    on=USER_ID_COL,
    how="left"
)

merged_df["classification_transferred"] = merged_df[CLASSIFICATION_COL].notna()

# Optional: Spalte für spätere LLM-Klassifizierung
merged_df["needs_llm_classification"] = ~merged_df["classification_transferred"]


# =========================
# Summary auf User-Level
# =========================

user_level = (
    merged_df
    .drop_duplicates(subset=[USER_ID_COL])
    .copy()
)

summary_by_state = (
    user_level
    .groupby(STATE_COL)
    .agg(
        users_total=(USER_ID_COL, "nunique"),
        users_already_classified=("classification_transferred", "sum"),
        users_still_needed=("needs_llm_classification", "sum")
    )
    .reset_index()
)

summary_by_state["share_already_classified"] = (
    summary_by_state["users_already_classified"] /
    summary_by_state["users_total"]
)

summary_by_state["share_still_needed"] = (
    summary_by_state["users_still_needed"] /
    summary_by_state["users_total"]
)

summary_by_state = summary_by_state.sort_values(STATE_COL).reset_index(drop=True)


# =========================
# Zusätzlich Summary nach NCHS-Gruppe, falls vorhanden
# =========================

if "nchs_group" in user_level.columns:
    summary_by_state_nchs = (
        user_level
        .groupby([STATE_COL, "nchs_group"])
        .agg(
            users_total=(USER_ID_COL, "nunique"),
            users_already_classified=("classification_transferred", "sum"),
            users_still_needed=("needs_llm_classification", "sum")
        )
        .reset_index()
    )

    summary_by_state_nchs["share_already_classified"] = (
        summary_by_state_nchs["users_already_classified"] /
        summary_by_state_nchs["users_total"]
    )

    summary_by_state_nchs["share_still_needed"] = (
        summary_by_state_nchs["users_still_needed"] /
        summary_by_state_nchs["users_total"]
    )
else:
    summary_by_state_nchs = pd.DataFrame()


# =========================
# Dateien speichern
# =========================

merged_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

summary_by_state.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

if not summary_by_state_nchs.empty:
    SUMMARY_NCHS_FILE = OUTPUT_DIR / "classification_transfer_summary_by_state_nchs.csv"

    summary_by_state_nchs.to_csv(
        SUMMARY_NCHS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

print("\nFertig.")
print(f"Neue Datei mit übernommenen Klassifizierungen:")
print(OUTPUT_FILE)

print(f"\nSummary:")
print(SUMMARY_FILE)

if not summary_by_state_nchs.empty:
    print(f"\nSummary nach NCHS-Gruppe:")
    print(SUMMARY_NCHS_FILE)

print("\nGesamtübersicht:")
print(f"User in neuer Datei: {user_level[USER_ID_COL].nunique():,}")
print(f"Bereits klassifiziert übernommen: {user_level['classification_transferred'].sum():,}")
print(f"Noch zu klassifizieren: {user_level['needs_llm_classification'].sum():,}")