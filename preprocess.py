import pandas as pd
from config import CLEANED_DATA_PATH, DATE_COLUMN

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw input DataFrame by handling missing values, parsing dates,
    and engineering relevant time-based features.
    """
    # Drop rows with null demand
    df = df.dropna(subset=["Demand"])

    # Convert date column to datetime if not already
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df = df.dropna(subset=[DATE_COLUMN])  # Drop rows where date parsing failed

    # Fill missing weather data with forward fill, then backward fill as fallback
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Feature Engineering: Add day, month, weekday, etc.
    df["Hour"] = df[DATE_COLUMN].dt.hour
    df["Day"] = df[DATE_COLUMN].dt.day
    df["Month"] = df[DATE_COLUMN].dt.month
    df["Weekday"] = df[DATE_COLUMN].dt.weekday
    df["Is_Weekend"] = df["Weekday"].isin([5, 6]).astype(int)

    return df


def load_cleaned_data() -> pd.DataFrame:
    """
    Loads and returns the cleaned dataset from disk.
    """
    return pd.read_csv(CLEANED_DATA_PATH, parse_dates=[DATE_COLUMN])