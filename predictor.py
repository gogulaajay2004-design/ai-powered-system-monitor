import pickle

import pandas as pd

CSV_FILE = "system_metrics.csv"

# Load trained models and feature names
with open("models/cpu_model.pkl", "rb") as file:
    cpu_model = pickle.load(file)

with open("models/ram_model.pkl", "rb") as file:
    ram_model = pickle.load(file)

with open("models/features.pkl", "rb") as file:
    features = pickle.load(file)

data = pd.read_csv(CSV_FILE)

if len(data) < 12:
    print("Prediction kosam minimum 12 readings kavali.")
    raise SystemExit

for column in ["cpu", "ram", "disk"]:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data = data.dropna(subset=["cpu", "ram", "disk"])

latest = data.iloc[-1]

input_data = {
    "cpu": latest["cpu"],
    "ram": latest["ram"],
    "disk": latest["disk"],

    "cpu_lag_1": data["cpu"].iloc[-2],
    "cpu_lag_3": data["cpu"].iloc[-4],
    "cpu_lag_6": data["cpu"].iloc[-7],
    "cpu_lag_12": data["cpu"].iloc[-12],

    "ram_lag_1": data["ram"].iloc[-2],
    "ram_lag_3": data["ram"].iloc[-4],
    "ram_lag_6": data["ram"].iloc[-7],
    "ram_lag_12": data["ram"].iloc[-12],

    "cpu_avg_30sec": data["cpu"].tail(6).mean(),
    "ram_avg_30sec": data["ram"].tail(6).mean(),

    "cpu_avg_1min": data["cpu"].tail(12).mean(),
    "ram_avg_1min": data["ram"].tail(12).mean()
}

model_input = pd.DataFrame(
    [input_data],
    columns=features
)

predicted_cpu = cpu_model.predict(model_input)[0]
predicted_ram = ram_model.predict(model_input)[0]

# Keep predictions within percentage limits
predicted_cpu = max(0, min(100, predicted_cpu))
predicted_ram = max(0, min(100, predicted_ram))

print("\n----- AI SYSTEM PREDICTION -----")
print(f"Current CPU Usage            : {latest['cpu']:.2f}%")
print(f"Predicted CPU after 5 minutes: {predicted_cpu:.2f}%")
print(f"Current RAM Usage            : {latest['ram']:.2f}%")
print(f"Predicted RAM after 5 minutes: {predicted_ram:.2f}%")

if predicted_cpu >= 85 or predicted_ram >= 85:
    print("Status: CRITICAL - High resource usage expected!")
elif predicted_cpu >= 70 or predicted_ram >= 75:
    print("Status: WARNING - Resource usage may become high.")
else:
    print("Status: HEALTHY - System usage is expected to remain normal.")