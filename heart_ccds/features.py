"""Step 2: Preprocessing pipeline setup, feature definitions, and data splitting.

Usage from CLI:
    python -m heart_ccds.features
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from heart_ccds.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
    TARGET,
    logger,
)
from heart_ccds.dataset import clean_data, load_data


def build_preprocessor(
    numeric_features = None,
    categorical_features = None,
):
    """Build the ColumnTransformer for numeric and categorical preprocessing."""
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    numeric_transformer = PowerTransformer()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor


def split_data(
    df,
    target,
    test_size,
    random_state,
):
    """Split dataframe into feature matrices and target vectors."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not present in dataframe.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Split data into train: {X_train.shape} and test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def prepare_and_save_splits(
    df,
    test_size,
    random_state,
    train_path,
    test_path,
):
    """Split dataset into train and test DataFrames and save them to CSV."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET] if TARGET in df.columns else None,
    )

    t_path = Path(train_path)
    te_path = Path(test_path)
    t_path.parent.mkdir(parents=True, exist_ok=True)
    te_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(t_path, index=False)
    test_df.to_csv(te_path, index=False)

    logger.info(f"Saved train set to {t_path} ({len(train_df)} rows)")
    logger.info(f"Saved test set to {te_path} ({len(test_df)} rows)")

    return train_df, test_df


def main():
    """CLI entrypoint for preprocessing and generating train/test splits."""
    parser = argparse.ArgumentParser(description="Build and save train and test splits.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(RAW_DATA_PATH),
        help=f"Path to input CSV file (default: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Test split fraction (default: {DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random state seed (default: {DEFAULT_RANDOM_STATE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help=f"Directory to save train.csv and test.csv (default: {PROCESSED_DATA_DIR})",
    )
    args = parser.parse_args()

    df = clean_data(load_data(args.data))
    out_dir = Path(args.output_dir)
    prepare_and_save_splits(
        df=df,
        test_size=args.test_size,
        random_state=args.random_state,
        train_path=out_dir / "train.csv",
        test_path=out_dir / "test.csv",
    )


if __name__ == "__main__":
    main()
