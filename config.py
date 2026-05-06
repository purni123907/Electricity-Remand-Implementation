# ===============================
# APP CONFIGURATION
# ===============================

APP_NAME = "Energy AI System"
THEME = "dark"
VERSION = "1.0.0"

# ===============================
# MODEL CONFIG
# ===============================

MODEL_TYPE = "RandomForest"

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

# ===============================
# FORECAST CONFIG
# ===============================

FORECAST_DAYS = 5

# ===============================
# UI CONFIG
# ===============================

DEFAULT_TEMPERATURE = 30
DEFAULT_HOUR = 12
DEFAULT_HOUSEHOLD_SIZE = 4
DEFAULT_MONTH = 6

# ===============================
# API CONFIG
# ===============================

OPENAI_MODEL = "gpt-4o-mini"
API_KEY = "YOUR_OPENAI_API_KEY"

# ===============================
# PATHS
# ===============================

DATA_PATH = "data/smart_home_energy_consumption_large.csv"
MODEL_PATH = "models/trained_model.pkl"
LOG_PATH = "logs/logs.txt"