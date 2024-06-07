import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取basis_sleep数据
    timestamps_sleep_begin = []
    timestamps_sleep_end = []
    sleep_quality = []

    for event in root.find("basis_sleep"):
        timestamps_sleep_begin.append(
            datetime.strptime(event.get("tbegin"), "%d-%m-%Y %H:%M:%S")
        )
        timestamps_sleep_end.append(
            datetime.strptime(event.get("tend"), "%d-%m-%Y %H:%M:%S")
        )
        sleep_quality.append(int(event.get("quality")))

    # 创建DataFrame
    df_sleep = pd.DataFrame(
        {
            "Sleep Begin": timestamps_sleep_begin,
            "Sleep End": timestamps_sleep_end,
            "Sleep Quality": sleep_quality,
        }
    )

    # 可视化basis_sleep数据
    plt.figure(figsize=(12, 6))

    for index, row in df_sleep.iterrows():
        plt.hlines(
            y=index,
            xmin=row["Sleep Begin"],
            xmax=row["Sleep End"],
            colors="blue",
            label="Sleep" if index == 0 else "",
        )

    plt.title("Sleep Periods Over Time")
    plt.xlabel("Time")
    plt.ylabel("Sleep Sessions")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
