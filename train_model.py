import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

CSV_FILE = "system_metrics.csv"
PREDICTION_STEPS = 60  # 60 readings × approximately 5 sec = 5 minutes

data = pd.read_csv(CSV_FILE)

required_columns = ["cpu", "ram", "disk"]

for column in required_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Previous system readings as model features
for lag in [1, 3, 6, 12]:
    data[f"cpu_lag_{lag}"] = data["cpu"].shift(lag)
    data[f"ram_lag_{lag}"] = data["ram"].shift(lag)

# Recent average usage
data["cpu_avg_30sec"] = data["cpu"].rolling(6).mean()
data["ram_avg_30sec"] = data["ram"].rolling(6).mean()

data["cpu_avg_1min"] = data["cpu"].rolling(12).mean()
data["ram_avg_1min"] = data["ram"].rolling(12).mean()

# Values approximately five minutes into the future
data["future_cpu"] = data["cpu"].shift(-PREDICTION_STEPS)
data["future_ram"] = data["ram"].shift(-PREDICTION_STEPS)

data = data.dropna().reset_index(drop=True)

features = [
    "cpu",
    "ram",
    "disk",
    "cpu_lag_1",
    "cpu_lag_3",
    "cpu_lag_6",
    "cpu_lag_12",
    "ram_lag_1",
    "ram_lag_3",
    "ram_lag_6",
    "ram_lag_12",
    "cpu_avg_30sec",
    "ram_avg_30sec",
    "cpu_avg_1min",
    "ram_avg_1min"
]

X = data[features]
y_cpu = data["future_cpu"]
y_ram = data["future_ram"]

# Time-based train/test split
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

cpu_train = y_cpu.iloc[:split_index]
cpu_test = y_cpu.iloc[split_index:]

ram_train = y_ram.iloc[:split_index]
ram_test = y_ram.iloc[split_index:]

cpu_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

ram_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

cpu_model.fit(X_train, cpu_train)
ram_model.fit(X_train, ram_train)

cpu_predictions = cpu_model.predict(X_test)
ram_predictions = ram_model.predict(X_test)

cpu_error = mean_absolute_error(cpu_test, cpu_predictions)
ram_error = mean_absolute_error(ram_test, ram_predictions)

os.makedirs("models", exist_ok=True)

with open("models/cpu_model.pkl", "wb") as file:
    pickle.dump(cpu_model, file)

with open("models/ram_model.pkl", "wb") as file:
    pickle.dump(ram_model, file)

with open("models/features.pkl", "wb") as file:
    pickle.dump(features, file)

print("AI models trained successfully!")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"CPU Mean Absolute Error: {cpu_error:.2f}%")
print(f"RAM Mean Absolute Error: {ram_error:.2f}%")
print("Models saved inside the models folder.")