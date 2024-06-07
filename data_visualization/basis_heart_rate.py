import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取basis_heart_rate数据
    timestamps_heart_rate = []
    heart_rate_values = []

    for event in root.find("basis_heart_rate"):
        timestamps_heart_rate.append(
            datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S")
        )
        heart_rate_values.append(int(event.get("value")))

    # 创建DataFrame
    df_heart_rate = pd.DataFrame(
        {"Timestamp": timestamps_heart_rate, "Heart Rate": heart_rate_values}
    )

    # 可视化basis_heart_rate数据
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_heart_rate["Timestamp"],
        df_heart_rate["Heart Rate"],
        color="red",
        label="Heart Rate",
    )
    plt.title("Heart Rate Over Time")
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (BPM)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
