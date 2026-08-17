"""
05_4_check_csv_schema_consistency.py

Purpose:
--------
Checks whether all CSV files in a folder have the same data format.

The script checks:
- readable files
- empty files
- column names
- column order
- missing columns
- extra columns
- inferred data types
- row counts
- important column validity

Input:
------
data/processed/05_sampling/05_2c new data/

Output:
-------
data/processed/05_sampling/05_2c_schema_check/
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2c new data"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2c_schema_check"
)

SUMMARY_REPORT_FILE = OUTPUT_DIR / "csv_schema_summary_report.csv"
COLUMN_COMPARISON_FILE = OUTPUT_DIR / "csv_column_comparison_report.csv"
DTYPE_REPORT_FILE = OUTPUT_DIR / "csv_dtype_report.csv"
VALUE_QUALITY_REPORT_FILE = OUTPUT_DIR / "csv_value_quality_report.csv"


# Passe diese Spalten ggf. an deine echten Spaltennamen an
IMPORTANT_COLUMNS = [
    "id",
    "date",
    "user_id",
    "user_state",
    "user_city",
    "nchs_code",
]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def find_csv_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    Reads a CSV safely as strings first.
    This avoids dtype conflicts across files.
    """
    return pd.read_csv(
        file_path,
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig"
    )


def infer_column_types(df: pd.DataFrame) -> dict:
    """
    Infers simple semantic data types per column.
    This is better than relying only on pandas dtype=str.
    """
    result = {}

    for col in df.columns:
        series = df[col].dropna().astype(str).str.strip()

        if series.empty:
            result[col] = "empty"
            continue

        sample = series.head(1000)

        numeric_rate = pd.to_numeric(sample, errors="coerce").notna().mean()
        datetime_rate = pd.to_datetime(sample, errors="coerce", utc=True).notna().mean()

        if numeric_rate >= 0.95:
            result[col] = "numeric_like"
        elif datetime_rate >= 0.95:
            result[col] = "datetime_like"
        else:
            result[col] = "text_like"

    return result


def check_value_quality(df: pd.DataFrame, file_name: str) -> dict:
    result = {
        "file": file_name,
        "rows": len(df),
    }

    for col in IMPORTANT_COLUMNS:
        if col not in df.columns:
            result[f"{col}_exists"] = False
            result[f"{col}_missing_values"] = None
            result[f"{col}_empty_values"] = None
            result[f"{col}_invalid_values"] = None
            continue

        series = df[col]

        result[f"{col}_exists"] = True
        result[f"{col}_missing_values"] = series.isna().sum()
        result[f"{col}_empty_values"] = (
            series.astype(str).str.strip() == ""
        ).sum()

        if col == "date":
            invalid = pd.to_datetime(
                series,
                errors="coerce",
                utc=True
            ).isna().sum()

            result[f"{col}_invalid_values"] = invalid

        elif col == "nchs_code":
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = (~numeric.isin([1, 2, 3, 4, 5, 6])).sum()

            result[f"{col}_invalid_values"] = invalid

        elif col == "user_state":
            valid_state_like = (
                series.astype(str)
                .str.strip()
                .str.upper()
                .str.match(r"^[A-Z]{2}$", na=False)
            )

            result[f"{col}_invalid_values"] = (~valid_state_like).sum()

        else:
            result[f"{col}_invalid_values"] = None

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_csv_files(INPUT_DIR)

    print(f"CSV files found: {len(files):,}")

    summary_rows = []
    column_comparison_rows = []
    dtype_rows = []
    value_quality_rows = []

    reference_columns = None
    reference_file = None
    reference_dtype_map = None

    for file_path in tqdm(files, desc="Checking CSV files"):
        try:
            df = read_csv_safely(file_path)

            rows = len(df)
            columns = list(df.columns)

            if reference_columns is None:
                reference_columns = columns
                reference_file = file_path.name
                reference_dtype_map = infer_column_types(df)

            missing_columns = [
                col for col in reference_columns
                if col not in columns
            ]

            extra_columns = [
                col for col in columns
                if col not in reference_columns
            ]

            same_columns_set = set(columns) == set(reference_columns)
            same_column_order = columns == reference_columns

            dtype_map = infer_column_types(df)

            dtype_differences = []

            for col in reference_columns:
                ref_type = reference_dtype_map.get(col)
                current_type = dtype_map.get(col)

                if ref_type != current_type:
                    dtype_differences.append(
                        f"{col}: reference={ref_type}, current={current_type}"
                    )

            summary_rows.append(
                {
                    "file": file_path.name,
                    "status": "checked",
                    "rows": rows,
                    "columns_count": len(columns),
                    "same_columns_set_as_reference": same_columns_set,
                    "same_column_order_as_reference": same_column_order,
                    "missing_columns_count": len(missing_columns),
                    "extra_columns_count": len(extra_columns),
                    "dtype_differences_count": len(dtype_differences),
                    "missing_columns": ", ".join(missing_columns),
                    "extra_columns": ", ".join(extra_columns),
                    "dtype_differences": " | ".join(dtype_differences),
                    "error": "",
                }
            )

            for col in sorted(set(reference_columns + columns)):
                column_comparison_rows.append(
                    {
                        "file": file_path.name,
                        "column": col,
                        "exists_in_file": col in columns,
                        "exists_in_reference": col in reference_columns,
                        "position_in_file": columns.index(col) + 1 if col in columns else None,
                        "position_in_reference": reference_columns.index(col) + 1 if col in reference_columns else None,
                    }
                )

            for col, dtype in dtype_map.items():
                dtype_rows.append(
                    {
                        "file": file_path.name,
                        "column": col,
                        "inferred_type": dtype,
                    }
                )

            value_quality_rows.append(
                check_value_quality(df, file_path.name)
            )

        except pd.errors.EmptyDataError:
            summary_rows.append(
                {
                    "file": file_path.name,
                    "status": "empty_file",
                    "rows": 0,
                    "columns_count": 0,
                    "same_columns_set_as_reference": False,
                    "same_column_order_as_reference": False,
                    "missing_columns_count": None,
                    "extra_columns_count": None,
                    "dtype_differences_count": None,
                    "missing_columns": "",
                    "extra_columns": "",
                    "dtype_differences": "",
                    "error": "EmptyDataError",
                }
            )

        except Exception as e:
            summary_rows.append(
                {
                    "file": file_path.name,
                    "status": "error",
                    "rows": 0,
                    "columns_count": 0,
                    "same_columns_set_as_reference": False,
                    "same_column_order_as_reference": False,
                    "missing_columns_count": None,
                    "extra_columns_count": None,
                    "dtype_differences_count": None,
                    "missing_columns": "",
                    "extra_columns": "",
                    "dtype_differences": "",
                    "error": str(e),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    column_comparison_df = pd.DataFrame(column_comparison_rows)
    dtype_df = pd.DataFrame(dtype_rows)
    value_quality_df = pd.DataFrame(value_quality_rows)

    summary_df.to_csv(
        SUMMARY_REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    column_comparison_df.to_csv(
        COLUMN_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    dtype_df.to_csv(
        DTYPE_REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    value_quality_df.to_csv(
        VALUE_QUALITY_REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Reference file: {reference_file}")
    print(f"Summary report: {SUMMARY_REPORT_FILE}")
    print(f"Column comparison report: {COLUMN_COMPARISON_FILE}")
    print(f"Dtype report: {DTYPE_REPORT_FILE}")
    print(f"Value quality report: {VALUE_QUALITY_REPORT_FILE}")

    if not summary_df.empty:
        files_with_issues = summary_df[
            (summary_df["status"] != "checked")
            | (summary_df["same_columns_set_as_reference"] == False)
            | (summary_df["same_column_order_as_reference"] == False)
            | (summary_df["missing_columns_count"].fillna(0) > 0)
            | (summary_df["extra_columns_count"].fillna(0) > 0)
            | (summary_df["dtype_differences_count"].fillna(0) > 0)
        ]

        print("\nSummary:")
        print(f"Files checked: {len(summary_df):,}")
        print(f"Files with issues: {len(files_with_issues):,}")


if __name__ == "__main__":
    main()