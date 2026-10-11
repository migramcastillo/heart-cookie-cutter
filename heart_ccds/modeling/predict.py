"""Step 4: Inference and Model Evaluation.

Usage from CLI:
    python -m heart_ccds.modeling.predict --model-uri runs:/<run_id>/model
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from heart_ccds.config import (
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    TARGET,
    TEST_DATA_PATH,
    logger,
)


def compute_metrics(y_true, y_pred):
    """Calculate classification performance metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        accuracy, precision, recall, and f1_score.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def load_model(model_uri):
    """Load logged or registered model from MLflow.

    Parameters
    ----------
    model_uri : str
        MLflow model URI (e.g. 'runs:/<run_id>/model' or 'models:/<name>/staging').

    Returns
    -------
    sklearn.pipeline.Pipeline
        Loaded model/pipeline.
    """
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Loading MLflow model from URI: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict(model, data):
    """Make predictions on incoming feature dataframe.

    Parameters
    ----------
    model : Pipeline or estimator
        Fitted model pipeline.
    data : pd.DataFrame
        Input features (if target column is present, it will be automatically dropped).

    Returns
    -------
    np.ndarray
        Array of predictions.
    """
    features = data.drop(columns=[TARGET], errors="ignore")
    return model.predict(features)


def main():
    """CLI entrypoint for inference and evaluation."""
    parser = argparse.ArgumentParser(description="Run inference and evaluate model using MLflow.")
    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="MLflow model URI (e.g., runs:/<run_id>/model or models:/rf-pipeline-heart/1)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(TEST_DATA_PATH),
        help=f"Path to input CSV data for predictions (default: {TEST_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPORTS_DIR / "predictions.csv"),
        help="Path to output predictions CSV (default: reports/predictions.csv)",
    )
    args = parser.parse_args()

    model = load_model(args.model_uri)
    df = pd.read_csv(args.data)
    logger.info(f"Loaded evaluation data from {args.data} ({len(df)} rows)")

    preds = predict(model, df)
    df_out = df.copy()
    df_out["prediction"] = preds

    # If ground truth target exists, compute and display evaluation metrics
    if TARGET in df.columns:
        metrics = compute_metrics(df[TARGET], preds)
        logger.info("Evaluation Metrics on provided data:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
