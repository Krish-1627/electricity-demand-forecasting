import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def save_model(model, filepath: str):
    """
    Saves the trained model to a pickle (.pkl) file.
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath: str):
    """
    Loads a model from a pickle (.pkl) file.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluates the model using Root Mean Squared Error (RMSE).
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def forecast_future(model, future_features: pd.DataFrame) -> pd.Series:
    """
    Forecasts future demand using the trained model.
    """
    return model.predict(future_features)