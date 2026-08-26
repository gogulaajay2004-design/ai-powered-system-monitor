import os
import pickle

import pandas as pd
import plotly.express as px
import psutil
import streamlit as st


CSV_FILE = "system_metrics.csv"

st.set_page_config(
    page_title="AI System Monitor",
    page_icon="🖥️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #172554 0%, #0f172a 35%, #020617 100%);
        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: "Segoe UI", sans-serif;
    }

    h1 {
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    [data-testid="stCaptionContainer"] {
        color: #94a3b8;
        font-size: 1rem;
    }

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.10);
    }

    [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 14px;
        overflow: hidden;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.25);
        transition: transform 0.2s ease;
    }

    .stDownloadButton > button:hover {
        color: white;
        border: none;
        transform: translateY(-2px);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖥️ AI-Powered Real-Time System Monitor")

st.caption(
    "Live monitoring, five-minute prediction and anomaly detection"
)


@st.cache_resource
def load_models():
    with open("models/cpu_model.pkl", "rb") as file:
        cpu_model = pickle.load(file)

    with open("models/ram_model.pkl", "rb") as file:
        ram_model = pickle.load(file)

    with open("models/features.pkl", "rb") as file:
        prediction_features = pickle.load(file)

    with open("models/anomaly_model.pkl", "rb") as file:
        anomaly_model = pickle.load(file)

    with open("models/anomaly_features.pkl", "rb") as file:
        anomaly_features = pickle.load(file)

    return (
        cpu_model,
        ram_model,
        prediction_features,
        anomaly_model,
        anomaly_features
    )


def create_prediction_input(data, features):
    latest = data.iloc[-1]

    values = {
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

    return pd.DataFrame(
        [values],
        columns=features
    )


def create_anomaly_input(data, features):
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    sent_change = max(
        0,
        latest["network_sent_mb"]
        - previous["network_sent_mb"]
    )

    received_change = max(
        0,
        latest["network_received_mb"]
        - previous["network_received_mb"]
    )

    values = {
        "cpu": latest["cpu"],
        "ram": latest["ram"],
        "disk": latest["disk"],
        "network_sent_change": sent_change,
        "network_received_change": received_change
    }

    return pd.DataFrame(
        [values],
        columns=features
    )


def get_top_processes():
    process_list = []

    for process in psutil.process_iter(
        ["pid", "name", "memory_info", "cpu_percent"]
    ):
        try:
            memory_info = process.info["memory_info"]

            if memory_info is None:
                continue

            memory_mb = (
                memory_info.rss / (1024 * 1024)
            )

            process_list.append({
                "PID": process.info["pid"],
                "Process": process.info["name"],
                "Memory (MB)": round(memory_mb, 2),
                "CPU (%)": process.info["cpu_percent"]
            })

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess
        ):
            continue

    processes = pd.DataFrame(process_list)

    if processes.empty:
        return processes

    return processes.sort_values(
        "Memory (MB)",
        ascending=False
    ).head(10)


@st.fragment(run_every=5)
def live_dashboard():
    if not os.path.exists(CSV_FILE):
        st.error(
            "system_metrics.csv not found. "
            "Run collector.py first."
        )
        return

    model_files = [
        "models/cpu_model.pkl",
        "models/ram_model.pkl",
        "models/features.pkl",
        "models/anomaly_model.pkl",
        "models/anomaly_features.pkl"
    ]

    if not all(
        os.path.exists(path)
        for path in model_files
    ):
        st.error(
            "One or more AI model files are missing."
        )
        return

    try:
        data = pd.read_csv(CSV_FILE)

    except Exception as error:
        st.error(f"Unable to read system data: {error}")
        return

    required_columns = [
        "timestamp",
        "cpu",
        "ram",
        "disk",
        "network_sent_mb",
        "network_received_mb"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        st.error(
            "Missing columns: "
            + ", ".join(missing_columns)
        )
        return

    numeric_columns = [
        "cpu",
        "ram",
        "disk",
        "network_sent_mb",
        "network_received_mb"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data = data.dropna(
        subset=required_columns
    )

    if len(data) < 12:
        st.warning(
            "Waiting for at least 12 readings..."
        )
        return

    latest = data.iloc[-1]

    try:
        (
            cpu_model,
            ram_model,
            prediction_features,
            anomaly_model,
            anomaly_features
        ) = load_models()

        prediction_input = create_prediction_input(
            data,
            prediction_features
        )

        predicted_cpu = cpu_model.predict(
            prediction_input
        )[0]

        predicted_ram = ram_model.predict(
            prediction_input
        )[0]

        anomaly_input = create_anomaly_input(
            data,
            anomaly_features
        )

        anomaly_prediction = anomaly_model.predict(
            anomaly_input
        )[0]

        anomaly_score = anomaly_model.decision_function(
            anomaly_input
        )[0]

    except Exception as error:
        st.error(f"AI processing failed: {error}")
        return

    predicted_cpu = max(
        0,
        min(100, predicted_cpu)
    )

    predicted_ram = max(
        0,
        min(100, predicted_ram)
    )

    anomaly_detected = anomaly_prediction == -1

    columns = st.columns(6)

    columns[0].metric(
        "Current CPU",
        f"{latest['cpu']:.1f}%"
    )

    columns[1].metric(
        "Predicted CPU",
        f"{predicted_cpu:.1f}%"
    )

    columns[2].metric(
        "Current RAM",
        f"{latest['ram']:.1f}%"
    )

    columns[3].metric(
        "Predicted RAM",
        f"{predicted_ram:.1f}%"
    )

    columns[4].metric(
        "Disk Usage",
        f"{latest['disk']:.1f}%"
    )

    columns[5].metric(
        "Anomaly Status",
        "Detected" if anomaly_detected else "Normal",
        help=f"Anomaly score: {anomaly_score:.4f}"
    )

    if anomaly_detected:
        st.error(
            "🚨 ANOMALY DETECTED: Current system behaviour "
            "is different from the learned normal pattern."
        )

    if predicted_cpu >= 85 or predicted_ram >= 85:
        st.error(
            "🔴 CRITICAL: High resource usage predicted!"
        )

    elif predicted_cpu >= 70 or predicted_ram >= 75:
        st.warning(
            "🟠 WARNING: Resource usage may become high."
        )

    else:
        st.success(
            "🟢 HEALTHY: Resource usage is expected "
            "to remain normal."
        )

    recent_data = data.tail(100)

    usage_chart = px.line(
        recent_data,
        x="timestamp",
        y=["cpu", "ram", "disk"],
        title="Live System Resource Usage",
        labels={
            "timestamp": "Time",
            "value": "Usage (%)",
            "variable": "Resource"
        }
    )

    usage_chart.update_layout(
        yaxis_range=[0, 100],
        legend_title_text="Resource",
        template="plotly_dark",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        font={"color": "#cbd5e1"},
        title_font={"size": 20, "color": "#f8fafc"},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="x unified"
    )

    st.plotly_chart(
        usage_chart,
        use_container_width=True
    )

    network_chart = px.line(
        recent_data,
        x="timestamp",
        y=[
            "network_sent_mb",
            "network_received_mb"
        ],
        title="Network Data Usage",
        labels={
            "timestamp": "Time",
            "value": "Data (MB)",
            "variable": "Network"
        }
    )

    network_chart.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        font={"color": "#cbd5e1"},
        title_font={"size": 20, "color": "#f8fafc"},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="x unified"
    )

    st.plotly_chart(
        network_chart,
        use_container_width=True
    )

    st.subheader(
        "🔍 Top Memory-Consuming Processes"
    )

    top_processes = get_top_processes()

    if top_processes.empty:
        st.info(
            "Process information is unavailable."
        )

    else:
        st.dataframe(
            top_processes,
            use_container_width=True,
            hide_index=True
        )

        highest_process = top_processes.iloc[0]

        st.subheader("💡 Smart Recommendations")

        if anomaly_detected:
            st.warning(
                "Unusual system behaviour was detected. "
                "Check recently opened applications, downloads "
                "and background processes."
            )

        if predicted_ram >= 85:
            st.error(
                f"RAM usage is critical. "
                f"{highest_process['Process']} is currently "
                f"the highest memory-consuming process at "
                f"{highest_process['Memory (MB)']:.2f} MB."
            )

            st.info(
                "Save your work and close unnecessary browser "
                "tabs, unused applications and duplicate servers."
            )

        elif predicted_ram >= 75:
            st.warning(
                "RAM usage may remain high. Close applications "
                "that you are not currently using."
            )

        elif predicted_cpu >= 70:
            st.warning(
                "High CPU usage is predicted. Check downloads, "
                "builds and background processing."
            )

        elif not anomaly_detected:
            st.success(
                "No immediate action is required. "
                "System resources appear stable."
            )
    st.subheader("📥 Download Monitoring Report")

    report_data = recent_data.copy()

    report_data["predicted_cpu_5min"] = round(
        predicted_cpu,
        2
    )

    report_data["predicted_ram_5min"] = round(
        predicted_ram,
        2
    )

    report_data["anomaly_status"] = (
        "Detected"
        if anomaly_detected
        else "Normal"
    )

    csv_report = report_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download System Health Report",
        data=csv_report,
        file_name="system_health_report.csv",
        mime="text/csv"
    )

    st.subheader("📋 Recent System Readings")

    st.dataframe(
        recent_data.tail(10).sort_values(
            "timestamp",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


live_dashboard()
