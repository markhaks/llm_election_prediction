"""
load_data.py

Handles ingestion of the raw election tweet dataset.

Features:
- discovers dataset files automatically
- reads compressed CSV files (.gz)
- previews and validates data structure

Purpose:
Provide a scalable and reproducible foundation for the data pipeline.
"""

from pathlib import Path
from typing import List

import pandas as pd


RAW_DATA_DIR = Path("data/01 raw/usc-x-24-us-election")


def find_data_files(input_dir: Path) -> List[Path]:
    """
    Finds all compressed CSV files in the raw data folder.
    """
    files = sorted(input_dir.rglob("*.gz"))

    if not files:
        raise FileNotFoundError(f"No .gz files found in {input_dir}")

    return files


def preview_file(file_path: Path, n_rows: int = 5) -> None:
    """
    Reads a small preview from one compressed CSV file.
    """
    print(f"Reading file: {file_path}")

    df = pd.read_csv(file_path, nrows=n_rows)

    print("\nColumns:")
    print(list(df.columns))

    print("\nPreview:")
    print(df.head(n_rows))


def main() -> None:
    files = find_data_files(RAW_DATA_DIR)

    print(f"Found {len(files)} .gz files.")

    first_file = files[0]
    preview_file(first_file)


if __name__ == "__main__":
    main()