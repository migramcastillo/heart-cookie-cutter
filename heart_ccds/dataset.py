"""Step 1: Data ingestion and basic dataset cleaning.

Usage from CLI:
    python -m heart_ccds.dataset
"""

import argparse
from pathlib import Path
import pandas as pd

from heart_ccds.config import (
    INTERIM_DATA_DIR,
    RAW_DATA_PATH,
    TARGET,
    logger,
)


def load_data(path = RAW_DATA_PATH):
    """Load the heart dataset from a CSV file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    logger.info(f"Loading raw data from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    return df


def clean_data(df):
    """Perform baseline validation and cleaning on the dataset."""
    cleaned = df.copy()

    cleaned.columns = [c.strip() for c in cleaned.columns]

    if TARGET not in cleaned.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset columns: {list(cleaned.columns)}")

    # Remove duplicates -- in this case there are no dups
    initial_len = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    dropped_dups = initial_len - len(cleaned)
    if dropped_dups > 0:
        logger.info(f"Dropped {dropped_dups} duplicate rows.")

    logger.info(f"Cleaned dataset shape: {cleaned.shape}")
    return cleaned


def main():
    """CLI entrypoint for data loading step."""
    parser = argparse.ArgumentParser(description="Load and clean raw heart disease data.")
    parser.add_argument(
        "--input-path",
        type=str,
        default=str(RAW_DATA_PATH),
        help=f"Path to input raw CSV file (default: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(INTERIM_DATA_DIR / "interim.csv"),
        help="Path to output interim CSV file (default: data/interim/interim.csv)",
    )
    args = parser.parse_args()

    df_raw = load_data(args.input_path)
    df_clean = clean_data(df_raw)

    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_file, index=False)
    logger.info(f"Saved cleaned interim dataset to {out_file} ({len(df_clean)} rows)")


if __name__ == "__main__":
    main()
