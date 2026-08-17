"""
09_unique_users_by_state_week_nchs.py

Purpose:
--------
Evaluates how many unique users tweeted in the weeks before the 2024 US election,
split by state and NCHS urban-rural category.

Output example:
---------------
state, weeks_before_election, unique_users_total,
category_1_unique_users, category_1_percent, ...

Definition:
-----------
week 1 = last 7 days before election
week 2 = last 14 days before election
week 3 = last 21 days before election
etc.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2b tweets_without_puerto_rico"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\06_analysis"
)

OUTPUT_FILE = OUTPUT_DIR / "unique_users_by_state_week_nchs.csv"
REPORT_FILE = OUTPUT_DIR / "unique_users_by_state_week_nchs_report.csv"


# US Election Day 2024
ELECTION_DATE = pd.Timestamp("2024-11-05")

# How many weeks before election should be evaluated?
MAX_WEEKS_BEFORE_ELECTION = 12


# Adjust these if your columns have different names
DATE_COLUMN = "tweet_date"
USER_ID_COLUMN = "user_id"
STATE_COLUMN = "user_state"
NCHS_CODE_COLUMN = "nchs_code"


NCHS_CODES = [1, 2, 3, 4, 5, 6]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def find_csv_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


def load_relevant_columns(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype=str)

    required_columns = [
        DATE_COLUMN,
        USER_ID_COLUMN,
        STATE_COLUMN,
        NCHS_CODE_COLUMN,
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {file_path.name}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required_columns].copy()

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

    df[USER_ID_COLUMN] = df[USER_ID_COLUMN].astype(str).str.strip()
    df[STATE_COLUMN] = df[STATE_COLUMN].astype(str).str.strip().str.upper()

    df[NCHS_CODE_COLUMN] = pd.to_numeric(
        df[NCHS_CODE_COLUMN],
        errors="coerce"
    ).astype("Int64")

    df = df.dropna(
        subset=[
            DATE_COLUMN,
            USER_ID_COLUMN,
            STATE_COLUMN,
            NCHS_CODE_COLUMN,
        ]
    )

    df = df[df[USER_ID_COLUMN] != ""]
    df = df[df[STATE_COLUMN] != ""]
    df = df[df[NCHS_CODE_COLUMN].isin(NCHS_CODES)]

    return df


def calculate_weekly_unique_users(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for week in range(1, MAX_WEEKS_BEFORE_ELECTION + 1):
        start_date = ELECTION_DATE - pd.Timedelta(days=7 * week)
        end_date = ELECTION_DATE

        week_df = df[
            (df[DATE_COLUMN] >= start_date)
            & (df[DATE_COLUMN] < end_date)
        ].copy()

        if week_df.empty:
            continue

        total_unique = (
            week_df
            .groupby(STATE_COLUMN)[USER_ID_COLUMN]
            .nunique()
            .reset_index(name="unique_users_total")
        )

        category_unique = (
            week_df
            .groupby([STATE_COLUMN, NCHS_CODE_COLUMN])[USER_ID_COLUMN]
            .nunique()
            .reset_index(name="unique_users")
        )

        pivot = category_unique.pivot(
            index=STATE_COLUMN,
            columns=NCHS_CODE_COLUMN,
            values="unique_users"
        ).fillna(0)

        for code in NCHS_CODES:
            if code not in pivot.columns:
                pivot[code] = 0

        pivot = pivot[NCHS_CODES]
        pivot = pivot.reset_index()

        pivot = pivot.merge(
            total_unique,
            on=STATE_COLUMN,
            how="left"
        )

        pivot["weeks_before_election"] = week
        pivot["start_date"] = start_date.date()
        pivot["end_date_exclusive"] = end_date.date()

        for code in NCHS_CODES:
            count_col = f"category_{code}_unique_users"
            percent_col = f"category_{code}_percent"

            pivot[count_col] = pivot[code].astype(int)

            pivot[percent_col] = (
                pivot[count_col]
                / pivot["unique_users_total"]
                * 100
            ).round(2)

        keep_columns = [
            STATE_COLUMN,
            "weeks_before_election",
            "start_date",
            "end_date_exclusive",
            "unique_users_total",
        ]

        for code in NCHS_CODES:
            keep_columns.append(f"category_{code}_unique_users")
            keep_columns.append(f"category_{code}_percent")

        results.append(pivot[keep_columns])

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_csv_files(INPUT_DIR)

    print(f"Found CSV files: {len(files):,}")

    all_data = []
    report_rows = []

    for file_path in tqdm(files, desc="Loading CSV files"):
        try:
            df = load_relevant_columns(file_path)

            all_data.append(df)

            report_rows.append(
                {
                    "file": file_path.name,
                    "status": "loaded",
                    "rows_loaded": len(df),
                    "error": "",
                }
            )

        except Exception as e:
            report_rows.append(
                {
                    "file": file_path.name,
                    "status": "error",
                    "rows_loaded": 0,
                    "error": str(e),
                }
            )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

    if not all_data:
        raise RuntimeError("No valid data could be loaded.")

    print("Combining data...")
    df_all = pd.concat(all_data, ignore_index=True)

    print(f"Rows used for analysis: {len(df_all):,}")
    print("Calculating unique users by state, week and NCHS category...")

    result_df = calculate_weekly_unique_users(df_all)

    result_df = result_df.sort_values(
        by=[STATE_COLUMN, "weeks_before_election"]
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Report file: {REPORT_FILE}")


if __name__ == "__main__":
    main()