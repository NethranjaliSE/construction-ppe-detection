from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_DIR = RAW_DIR / "construction-safety"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_YAML = DATA_DIR / "data.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42