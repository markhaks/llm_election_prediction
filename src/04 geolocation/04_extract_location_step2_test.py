"""
04_2_fast_strict_city_state_match.py

Purpose:
--------
Fast strict city/state matching for user_location.

It accepts mainly clear patterns like:
- Chicago, IL
- Chicago, Illinois
- Dallas TX
- Miami, Florida

It avoids broad substring matching like:
- England -> England, AR
- London, England -> London, KY

Input:
------
data/processed/04_extract_location/Step 1/
data/external/uscities.csv

Output:
-------
data/processed/04_extract_location/Step 2_fast_strict/
"""

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\04_extract_location\Step 1"
)

CITY_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\external\uscities.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\04_extract_location\Step 2_fast_strict"
)

MATCHED_DIR = OUTPUT_DIR / "city_state_matched"
NOT_MATCHED_DIR = OUTPUT_DIR / "city_state_not_matched"
REPORT_FILE = OUTPUT_DIR / "fast_strict_city_state_matching_report.csv"

INPUT_FILE_PATTERN = "*_location_filtered.csv"


STATE_NAME_TO_ID = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC",
}

STATE_IDS = set(STATE_NAME_TO_ID.values())

NON_US_TERMS = {
    "england", "united kingdom", "uk", "u k", "u.k.", "scotland", "wales",
    "ireland", "canada", "mexico", "france", "germany", "deutschland",
    "italy", "spain", "australia", "new zealand", "india", "china",
    "japan", "brazil", "argentina", "colombia", "nigeria", "kenya",
    "south africa", "europe", "asia", "africa", "worldwide"
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace(".", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_city(value):
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob(INPUT_FILE_PATTERN))

    if not files:
        raise FileNotFoundError(f"No input CSV files found in: {input_dir}")

    return files


def load_city_lookup(city_file: Path) -> pd.DataFrame:
    cities = pd.read_csv(city_file, dtype=str, low_memory=False)

    required = [
        "city",
        "city_ascii",
        "state_id",
        "state_name",
        "county_fips",
        "county_name",
        "population",
    ]

    missing = [col for col in required if col not in cities.columns]

    if missing:
        raise ValueError(f"uscities.csv missing columns: {missing}")

    rows = []

    for city_col in ["city", "city_ascii"]:
        temp = cities[required].copy()
        temp["city_key"] = temp[city_col].apply(normalize_city)
        rows.append(temp)

    city_df = pd.concat(rows, ignore_index=True)

    city_df["state_key"] = city_df["state_id"].astype(str).str.strip().str.upper()
    city_df["population_num"] = pd.to_numeric(
        city_df["population"],
        errors="coerce"
    ).fillna(0)

    city_df = city_df.sort_values(
        by=["city_key", "state_key", "population_num"],
        ascending=[True, True, False]
    )

    city_df = city_df.drop_duplicates(
        subset=["city_key", "state_key"],
        keep="first"
    )

    return city_df[
        [
            "city_key",
            "state_key",
            "city",
            "state_id",
            "state_name",
            "county_fips",
            "county_name",
            "population",
        ]
    ].copy()


def contains_non_us_term(location_key):
    for term in NON_US_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", location_key):
            return True
    return False


def extract_candidate_city_state(location):
    """
    Extracts candidates from clear patterns.

    Examples:
    - Chicago, IL
    - Chicago, Illinois
    - Dallas TX
    - Miami / Florida
    """
    loc = normalize_text(location)

    if not loc:
        return "", "", "empty_location"

    if contains_non_us_term(loc):
        return "", "", "blocked_non_us_term"

    # Normalize separators
    cleaned = loc
    cleaned = cleaned.replace("|", ",")
    cleaned = cleaned.replace("/", ",")
    cleaned = cleaned.replace(";", ",")
    cleaned = cleaned.replace("-", ",")

    parts = [p.strip() for p in cleaned.split(",") if p.strip()]

    candidates = []

    # Pattern: City, State
    if len(parts) >= 2:
        for i in range(len(parts) - 1):
            city_part = normalize_city(parts[i])
            state_part = normalize_text(parts[i + 1])

            state_key = None

            if state_part.upper() in STATE_IDS:
                state_key = state_part.upper()
            elif state_part in STATE_NAME_TO_ID:
                state_key = STATE_NAME_TO_ID[state_part]

            if city_part and state_key:
                candidates.append((city_part, state_key))

    # Pattern: City ST at end, e.g. "Dallas TX"
    match = re.search(r"(.+?)\s+([A-Z]{2})$", str(location).strip())

    if match:
        city_part = normalize_city(match.group(1))
        state_part = match.group(2).upper()

        if state_part in STATE_IDS:
            candidates.append((city_part, state_part))

    unique_candidates = list(set(candidates))

    if len(unique_candidates) == 1:
        return unique_candidates[0][0], unique_candidates[0][1], "candidate_found"

    if len(unique_candidates) > 1:
        return "", "", "multiple_candidates"

    return "", "", "no_clear_city_state_pattern"


def process_file(file_path: Path, city_lookup: pd.DataFrame) -> dict:
    df = pd.read_csv(file_path, dtype=str, low_memory=False)

    if "user_location" not in df.columns:
        return {
            "file": file_path.name,
            "status": "missing_user_location",
            "rows_before": len(df),
            "rows_matched": 0,
            "rows_not_matched": len(df),
            "error": "",
        }

    rows_before = len(df)

    extracted = df["user_location"].apply(extract_candidate_city_state)

    df["city_key"] = extracted.apply(lambda x: x[0])
    df["state_key"] = extracted.apply(lambda x: x[1])
    df["user_state_match_method"] = extracted.apply(lambda x: x[2])

    merged = df.merge(
        city_lookup,
        on=["city_key", "state_key"],
        how="left",
    )

    matched_mask = merged["city"].notna()

    merged.loc[matched_mask, "user_city"] = merged.loc[matched_mask, "city"]
    merged.loc[matched_mask, "user_state"] = merged.loc[matched_mask, "state_id"]
    merged.loc[matched_mask, "user_state_name"] = merged.loc[matched_mask, "state_name"]
    merged.loc[matched_mask, "user_county_fips"] = merged.loc[matched_mask, "county_fips"]
    merged.loc[matched_mask, "user_county_name"] = merged.loc[matched_mask, "county_name"]
    merged.loc[matched_mask, "user_city_population"] = merged.loc[matched_mask, "population"]
    merged.loc[matched_mask, "user_state_match_method"] = "fast_strict_city_state_match"

    matched_df = merged[matched_mask].copy()
    not_matched_df = merged[~matched_mask].copy()

    drop_cols = [
        "city_key",
        "state_key",
        "city",
        "state_id",
        "state_name",
        "county_fips",
        "county_name",
        "population",
    ]

    matched_df = matched_df.drop(columns=drop_cols, errors="ignore")
    not_matched_df = not_matched_df.drop(columns=drop_cols, errors="ignore")

    matched_output = MATCHED_DIR / file_path.name.replace(
        "_location_filtered.csv",
        "_fast_strict_city_state_matched.csv"
    )

    not_matched_output = NOT_MATCHED_DIR / file_path.name.replace(
        "_location_filtered.csv",
        "_fast_strict_city_state_not_matched.csv"
    )

    matched_df.to_csv(matched_output, index=False, encoding="utf-8-sig")
    not_matched_df.to_csv(not_matched_output, index=False, encoding="utf-8-sig")

    return {
        "file": file_path.name,
        "status": "success",
        "rows_before": rows_before,
        "rows_matched": len(matched_df),
        "rows_not_matched": len(not_matched_df),
        "error": "",
    }


def main():
    MATCHED_DIR.mkdir(parents=True, exist_ok=True)
    NOT_MATCHED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading city lookup...")
    city_lookup = load_city_lookup(CITY_FILE)

    files = find_input_files(INPUT_DIR)

    print(f"Files found: {len(files)}")
    print(f"City lookup rows: {len(city_lookup):,}")

    report_rows = []

    for file_path in tqdm(files, desc="Fast strict city/state matching"):
        try:
            result = process_file(file_path, city_lookup)
            report_rows.append(result)

        except Exception as e:
            report_rows.append(
                {
                    "file": file_path.name,
                    "status": "error",
                    "rows_before": 0,
                    "rows_matched": 0,
                    "rows_not_matched": 0,
                    "error": str(e),
                }
            )

    report = pd.DataFrame(report_rows)
    report.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Matched folder: {MATCHED_DIR}")
    print(f"Not matched folder: {NOT_MATCHED_DIR}")
    print(f"Report: {REPORT_FILE}")

    total = report["rows_before"].sum()
    matched = report["rows_matched"].sum()

    if total > 0:
        print(f"Match rate: {matched / total:.2%}")


if __name__ == "__main__":
    main()