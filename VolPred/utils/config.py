"""Configuration for background tasks and scheduler."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)

# API Configuration
API_FUNCTION = "BRENT"
API_INTERVAL = "daily"
API_OUTPUTSIZE = "full"
API_KEY = os.getenv("API_KEY")

# Model files
MODEL_JSON_FILE = MODEL_DIR / f"{API_FUNCTION}_{API_INTERVAL}.json"
MODEL_JOBLIB_FILE = MODEL_DIR / f"{API_FUNCTION}_{API_INTERVAL}_garch.joblib"

# Scheduler Configuration
# Daily prediction task
DAILY_PREDICTION_HOUR = 18  # UTC hour (18:00 UTC)
DAILY_PREDICTION_MINUTE = 0

# Retraining task - every 10 days
RETRAIN_INTERVAL_DAYS = 10
RETRAIN_HOUR = 2  # Run retrain at 02:00 UTC to avoid market hours

# Model configuration
GARCH_P = 1
GARCH_Q = 1
TRAIN_TEST_SPLIT = 0.2

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "tasks.log"
LOG_LEVEL = "INFO"

# Metadata tracking
METADATA_FILE = MODEL_DIR / "retraining_metadata.json"
