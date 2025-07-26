import pandas as pd
import os

DEFAULT_CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"

def load_raw_data(filepath):
    """
    Loads raw CSV data from a specified path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")
    return pd.read_csv(filepath)

def load_cleaned_data(filepath=DEFAULT_CLEANED_DATA_PATH):
    """
    Loads cleaned data from the default or specified path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Cleaned data not found: {filepath}")
    return pd.read_csv(filepath, parse_dates=["Date"])