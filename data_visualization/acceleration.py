import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取acceleration数据
    timestamps_acceleration = []
    acceleration_values = []

    for event in root.find("acceleration"):
        timestamps_acceleration.append(
            datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S")
        )
        acceleration_values.append(float(event.get("value")))

    # 创建DataFrame
    df_acceleration = pd.DataFrame(
        {"Timestamp": timestamps_acceleration, "Acceleration": acceleration_values}
    )

    # 可视化acceleration数据
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_acceleration["Timestamp"],
        df_acceleration["Acceleration"],
        color="green",
        label="Acceleration",
    )
    plt.title("Acceleration Over Time")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (units)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
