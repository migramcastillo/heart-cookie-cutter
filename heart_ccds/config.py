from pathlib import Path

import os
import logging
from dotenv import load_dotenv

PROJ_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger("heart_ccds")

load_dotenv()

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"

RAW_DATA_PATH = RAW_DATA_DIR / "heart.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

TARGET = "HeartDisease"

NUMERIC_FEATURES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "ChestPainType",
    "FastingBS",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = None
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1

DEFAULT_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rf-classifier-pipeline-heart")
DEFAULT_MODEL_NAME = "rf-pipeline-heart"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
