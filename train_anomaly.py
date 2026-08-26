import os
import pickle

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CSV_FILE = "system_metrics.csv"

data = pd.read_csv(CSV_FILE)

required_columns = [
    "cpu",
    "ram",
    "disk",
    "network_sent_mb",
    "network_received_mb"
]

for column in required_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

# Network totals-ni speed/change values-ga convert chestunnam
data["network_sent_change"] = (
    data["network_sent_mb"]
    .diff()
    .clip(lower=0)
    .fillna(0)
)

data["network_received_change"] = (
    data["network_received_mb"]
    .diff()
    .clip(lower=0)
    .fillna(0)
)

features = [
    "cpu",
    "ram",
    "disk",
    "network_sent_change",
    "network_received_change"
]

training_data = data[features].dropna()

if len(training_data) < 100:
    print("Minimum 100 readings are required.")
    raise SystemExit

anomaly_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "isolation_forest",
        IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )
    )
])

anomaly_model.fit(training_data)

predictions = anomaly_model.predict(
    training_data
)

anomaly_count = int(
    (predictions == -1).sum()
)

normal_count = int(
    (predictions == 1).sum()
)

os.makedirs("models", exist_ok=True)

with open(
    "models/anomaly_model.pkl",
    "wb"
) as file:
    pickle.dump(anomaly_model, file)

with open(
    "models/anomaly_features.pkl",
    "wb"
) as file:
    pickle.dump(features, file)

print("Anomaly detection model trained successfully!")
print(f"Total readings: {len(training_data)}")
print(f"Normal readings: {normal_count}")
print(f"Anomalies detected: {anomaly_count}")
print("Model saved as models/anomaly_model.pkl")