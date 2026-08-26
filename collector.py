import csv
import os
import time
from datetime import datetime

import psutil

FILE_NAME = "system_metrics.csv"
file_exists = os.path.exists(FILE_NAME)

with open(FILE_NAME, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "cpu",
            "ram",
            "disk",
            "network_sent_mb",
            "network_received_mb"
        ])

    print("Real-Time System Monitoring started...")
    print("Stop cheyyadaniki Ctrl + C press cheyyi.\n")

    try:
        while True:
            network = psutil.net_io_counters()

            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage("C:\\").percent
            sent = round(network.bytes_sent / (1024 * 1024), 2)
            received = round(network.bytes_recv / (1024 * 1024), 2)

            writer.writerow([
                datetime.now().isoformat(),
                cpu,
                ram,
                disk,
                sent,
                received
            ])

            file.flush()

            print(
                f"CPU: {cpu}% | RAM: {ram}% | Disk: {disk}% | "
                f"Sent: {sent} MB | Received: {received} MB"
            )

            time.sleep(4)

    except KeyboardInterrupt:
        print("\nSystem monitoring stopped.")