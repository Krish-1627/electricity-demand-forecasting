from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares the raw electricity demand data.

    Steps:
    - Drop missing values
    - Convert 'Date' column to datetime
    - Sort data by date
    - Reset index

    Parameters:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned and prepared DataFrame.
    """
    df = df.copy()
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notnull()]
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def split_data(
    df: pd.DataFrame,
    target_column: str = "Demand",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataframe into train and test sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.

    Returns:
        Tuple containing:
        - X_train (pd.DataFrame)
        - X_test (pd.DataFrame)
        - y_train (pd.Series)
        - y_test (pd.Series)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)