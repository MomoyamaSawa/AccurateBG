import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取work数据
    timestamps_work_begin = []
    timestamps_work_end = []
    work_intensity = []

    for event in root.find("work"):
        timestamps_work_begin.append(
            datetime.strptime(event.get("ts_begin"), "%d-%m-%Y %H:%M:%S")
        )
        timestamps_work_end.append(
            datetime.strptime(event.get("ts_end"), "%d-%m-%Y %H:%M:%S")
        )
        work_intensity.append(int(event.get("intensity")))

    # 创建DataFrame
    df_work = pd.DataFrame(
        {
            "Work Begin": timestamps_work_begin,
            "Work End": timestamps_work_end,
            "Work Intensity": work_intensity,
        }
    )

    # 可视化work数据
    plt.figure(figsize=(12, 6))

    for index, row in df_work.iterrows():
        plt.plot(
            [row["Work Begin"], row["Work End"]],
            [row["Work Intensity"], row["Work Intensity"]],
            label="Work" if index == 0 else "",
            color="purple",
        )

    plt.title("Work Intensity Over Time")
    plt.xlabel("Time")
    plt.ylabel("Work Intensity")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
