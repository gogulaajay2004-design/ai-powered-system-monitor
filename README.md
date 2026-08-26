# AI-Powered Real-Time System Monitor

An intelligent system-monitoring application developed using Python and Machine Learning. It monitors system resources in real time, predicts CPU and RAM usage five minutes ahead, detects unusual behaviour and provides smart recommendations through an interactive dashboard.

## Features

- Real-time CPU monitoring
- Real-time RAM and disk monitoring
- Network usage tracking
- Five-minute CPU usage prediction
- Five-minute RAM usage prediction
- AI-based anomaly detection
- Healthy, warning and critical alerts
- Top memory-consuming process detection
- Smart system-health recommendations
- Interactive Plotly charts
- Downloadable CSV health report
- Automatic dashboard refresh

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Isolation Forest
- Streamlit
- Plotly
- Psutil

## Project Structure

```text
ai-system-monitor/
├── models/
│   ├── cpu_model.pkl
│   ├── ram_model.pkl
│   ├── features.pkl
│   ├── anomaly_model.pkl
│   └── anomaly_features.pkl
├── collector.py
├── dashboard.py
├── predictor.py
├── train_model.py
├── train_anomaly.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. `collector.py` collects CPU, RAM, disk and network readings.
2. The readings are stored in `system_metrics.csv`.
3. `train_model.py` trains Random Forest models for future CPU and RAM prediction.
4. `train_anomaly.py` trains an Isolation Forest model to identify unusual system behaviour.
5. `dashboard.py` displays live metrics, predictions, anomalies, processes and recommendations.
6. Users can download monitoring data as a CSV health report.

## Machine-Learning Models

### Resource Prediction

Random Forest Regression predicts CPU and RAM usage approximately five minutes into the future using:

- Current CPU, RAM and disk usage
- Previous CPU and RAM readings
- 30-second rolling averages
- One-minute rolling averages

### Anomaly Detection

Isolation Forest identifies system behaviour that differs from patterns learned during training. It analyses:

- CPU usage
- RAM usage
- Disk usage
- Change in sent network data
- Change in received network data

## Model Evaluation

The initial model produced the following results:

| Metric | Result |
|---|---:|
| CPU Mean Absolute Error | 7.93% |
| RAM Mean Absolute Error | 1.34% |
| Anomaly training readings | 435 |
| Potential anomaly candidates | 22 |

These results are based on data collected from a single local system. Accuracy can improve with more diverse and longer-duration training data.

## Installation

Clone the repository:

```bash
git clone https://github.com/gogulaajay2004-design/ai-powered-system-monitor.git
cd ai-system-monitor
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. Collect system data

```bash
python collector.py
```

Keep the collector running while using the dashboard.

### 2. Start the dashboard

Open a second terminal and run:

```bash
python -m streamlit run dashboard.py
```

Open the following address if the browser does not open automatically:

```text
http://localhost:8501
```

The repository includes trained model files. Retraining is optional and requires sufficient collected readings.

### 3. Retrain prediction models (optional)

```bash
python train_model.py
```

### 4. Retrain anomaly-detection model (optional)

```bash
python train_anomaly.py
```

## Dashboard Status

- **Healthy:** Resource usage is expected to remain normal.
- **Warning:** CPU or RAM usage may become high.
- **Critical:** Very high resource usage is predicted.
- **Normal anomaly status:** Current behaviour matches learned patterns.
- **Anomaly detected:** Current behaviour differs from learned patterns.

High usage and anomaly detection are separate signals. A system can have high but familiar usage, or normal overall usage with an unusual short-term pattern.

## Future Enhancements

- Email and Telegram alerts
- PDF health reports
- Multiple-device monitoring
- SQLite or cloud data storage
- LSTM time-series forecasting
- User authentication
- Docker deployment
- Automatic model retraining
- Historical daily and weekly reports

## Author

**Gogula Ajay**

- B.Tech Computer Science and Engineering
- Aspiring Data Scientist, AI/ML Developer and Python Developer
