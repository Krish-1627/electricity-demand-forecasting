import pandas as pd

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features like day, month, weekday, and hour from the Date column.
    """
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.weekday
    df["Hour"] = df["Date"].dt.hour
    return df

def normalize_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the 'Temperature' column using z-score if it exists.
    """
    if "Temperature" in df.columns:
        df["Temperature"] = (df["Temperature"] - df["Temperature"].mean()) / df["Temperature"].std()
    return df