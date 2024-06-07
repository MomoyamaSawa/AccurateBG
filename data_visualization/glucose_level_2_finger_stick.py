import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取数据
    timestamps_glucose = []
    glucose_values = []
    timestamps_finger_stick = []
    finger_stick_values = []

    for event in root.find("glucose_level"):
        timestamps_glucose.append(
            datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S")
        )
        glucose_values.append(int(event.get("value")))

    for event in root.find("finger_stick"):
        timestamps_finger_stick.append(
            datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S")
        )
        finger_stick_values.append(int(event.get("value")))

    # 创建DataFrame
    df_glucose = pd.DataFrame(
        {"Timestamp": timestamps_glucose, "Glucose Level": glucose_values}
    )
    df_finger_stick = pd.DataFrame(
        {"Timestamp": timestamps_finger_stick, "Finger Stick": finger_stick_values}
    )

    # 可视化
    plt.figure(figsize=(18, 6))
    plt.plot(
        df_glucose["Timestamp"], df_glucose["Glucose Level"], label="Glucose Level"
    )
    plt.scatter(
        df_finger_stick["Timestamp"],
        df_finger_stick["Finger Stick"],
        color="red",
        label="Finger Stick",
        marker="o",
    )
    plt.title("Glucose Level and Finger Stick Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
