# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
import numpy as np

# Load data
df = pd.read_csv("data/processed/cleaned_data.csv", parse_dates=["Date"])

#  Add datetime features
df["Hour"] = df["Date"].dt.hour
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month

# Define features and target
feature_cols = ["Temperature", "Humidity", "WindSpeed", "Hour", "DayOfWeek", "Month"]
X = df[feature_cols]
y = df["Demand"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, "models/lr_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")

# Print results
print(f" Linear Regression RMSE: {lr_rmse:.2f}")
print(f" XGBoost RMSE: {xgb_rmse:.2f}")