from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/01 raw")

files = sorted(RAW_DATA_DIR.rglob("*.gz"))

print(f"Found files: {len(files)}")

first_file = files[0]
df = pd.read_csv(first_file, nrows=5)

print(df.columns)

print(df.head(5).to_string())