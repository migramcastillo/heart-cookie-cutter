"""Step 3: Model training with MLflow tracking.

Usage from CLI:
    python -m heart_ccds.modeling.train
    python -m heart_ccds.modeling.train --n-estimators 200 --max-depth 10 --min-samples-split 4 --min-samples-leaf 2 --random-state 42
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from heart_ccds.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_MIN_SAMPLES_SPLIT,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    MLFLOW_TRACKING_URI,
    RAW_DATA_PATH,
    TARGET,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    logger,
)
from heart_ccds.dataset import clean_data, load_data
from heart_ccds.features import build_preprocessor, split_data
from heart_ccds.modeling.predict import compute_metrics


def build_pipeline(
    n_estimators = DEFAULT_N_ESTIMATORS,
    max_depth = DEFAULT_MAX_DEPTH,
    min_samples_split = DEFAULT_MIN_SAMPLES_SPLIT,
    min_samples_leaf = DEFAULT_MIN_SAMPLES_LEAF,
    random_state = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Construct full scikit-learn pipeline with preprocessor and RandomForestClassifier.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the random forest.
    max_depth : int or None
        Maximum depth of each tree.
    min_samples_split : int
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Unfitted full pipeline.
    """
    preprocessor = build_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_and_log(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators = DEFAULT_N_ESTIMATORS,
    max_depth = DEFAULT_MAX_DEPTH,
    min_samples_split = DEFAULT_MIN_SAMPLES_SPLIT,
    min_samples_leaf = DEFAULT_MIN_SAMPLES_LEAF,
    random_state = DEFAULT_RANDOM_STATE,
    experiment_name = DEFAULT_EXPERIMENT_NAME,
    run_name = None,
    register_model = False,
    model_name = DEFAULT_MODEL_NAME,
):
    """Train pipeline model and log parameters, metrics, and artifact to MLflow."""
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if not os.getenv("MLFLOW_RUN_ID") and not mlflow.active_run():
        mlflow.set_experiment(experiment_name)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
    }

    pipeline = build_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    with mlflow.start_run(run_name=run_name) as run:
        # Set descriptive tags
        mlflow.set_tag("model_family", "RandomForestClassifier")
        mlflow.set_tag("pipeline", "ColumnTransformer + PowerTransformer + OneHotEncoder")
        mlflow.set_tag("framework", "scikit-learn")

        logger.info(f"Training pipeline with params: {params}")
        pipeline.fit(X_train, y_train)

        # Predictions and metrics calculation
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        # MLflow logging
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log pipeline model artifact with input example
        input_example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"Logged Metrics: {metrics}")
        logger.info(f"Model URI: runs:/{run_id}/model")

        if register_model:
            model_uri = f"runs:/{run_id}/model"
            logger.info(f"Registering model '{model_name}' with URI {model_uri}")
            mlflow.register_model(model_uri=model_uri, name=model_name)

    return pipeline, metrics, run_id


def main():
    """CLI entrypoint for model training with configurable hyperparameters and seed."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier with MLflow tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default=str(RAW_DATA_PATH),
        help=f"Path to input CSV data (default: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proportion of data used for test set.",
    )

    # Hyperparameters (at least 3 hyperparameters + random seed)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Hyperparameter 1: Number of trees in Random Forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=lambda v: None if v in ("None", "none", "", "-1") else int(v),
        default=DEFAULT_MAX_DEPTH,
        help="Hyperparameter 2: Maximum depth of trees (int or None).",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=DEFAULT_MIN_SAMPLES_SPLIT,
        help="Hyperparameter 3: Minimum samples required to split an internal node.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=DEFAULT_MIN_SAMPLES_LEAF,
        help="Hyperparameter 4: Minimum samples required to be at a leaf node.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for data split and model training.",
    )

    # MLflow arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for the MLflow run.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register trained model in MLflow Model Registry.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Name for registered model if --register is set.",
    )

    args = parser.parse_args()

    # Determine whether args.data is raw data or train.csv
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Provided data file not found: {data_path}")

    # If processed train and test already exist and args.data points to train.csv:
    if data_path.name == "train.csv" and (data_path.parent / "test.csv").exists():
        logger.info(f"Loading pre-split train and test sets from: {data_path.parent}")
        train_df = pd.read_csv(data_path)
        test_df = pd.read_csv(data_path.parent / "test.csv")
        X_train = train_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]
        X_test = test_df.drop(columns=[TARGET])
        y_test = test_df[TARGET]
    else:
        # Load dataset and perform train/test split
        df = clean_data(load_data(data_path))
        X_train, X_test, y_train, y_test = split_data(
            df=df,
            target=TARGET,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    # Run training and MLflow logging
    pipeline, metrics, run_id = train_and_log(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        register_model=args.register,
        model_name=args.model_name,
    )

    print("\n" + "=" * 50)
    print("Training finished successfully!")
    print(f"MLflow Run ID: {run_id}")
    print(f"Model URI: runs:/{run_id}/model")
    print("Metrics:")
    for metric_name, val in metrics.items():
        print(f"  - {metric_name}: {val:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
