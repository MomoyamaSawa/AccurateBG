import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取exercise数据
    timestamps_exercise = []
    exercise_intensity = []
    exercise_duration = []

    for event in root.find("exercise"):
        timestamps_exercise.append(
            datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S")
        )
        exercise_intensity.append(int(event.get("intensity")))
        exercise_duration.append(int(event.get("duration")))

    # 创建DataFrame
    df_exercise = pd.DataFrame(
        {
            "Timestamp": timestamps_exercise,
            "Intensity": exercise_intensity,
            "Duration": exercise_duration,
        }
    )

    # 可视化exercise数据
    plt.figure(figsize=(12, 6))

    for index, row in df_exercise.iterrows():
        end_time = row["Timestamp"] + timedelta(minutes=row["Duration"])
        plt.plot(
            [row["Timestamp"], end_time],
            [row["Intensity"], row["Intensity"]],
            label="Exercise" if index == 0 else "",
            color="orange",
        )

    plt.title("Exercise Intensity and Duration Over Time")
    plt.xlabel("Time")
    plt.ylabel("Exercise Intensity")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
