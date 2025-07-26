import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# File paths
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_data.csv")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")

# Model options
MODEL_OPTIONS = {
    "Linear Regression": LR_MODEL_PATH,
    "XGBoost": XGB_MODEL_PATH,
}

# Date column
DATE_COLUMN = "Date"

# Target column
TARGET_COLUMN = "Demand"