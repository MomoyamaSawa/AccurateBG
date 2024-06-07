import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取sleep数据
    timestamps_sleep_begin = []
    timestamps_sleep_end = []
    sleep_quality = []

    for event in root.find("sleep"):
        timestamps_sleep_begin.append(
            datetime.strptime(event.get("ts_begin"), "%d-%m-%Y %H:%M:%S")
        )
        timestamps_sleep_end.append(
            datetime.strptime(event.get("ts_end"), "%d-%m-%Y %H:%M:%S")
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

    # 可视化sleep数据
    plt.figure(figsize=(12, 6))

    for index, row in df_sleep.iterrows():
        plt.plot(
            [row["Sleep Begin"], row["Sleep End"]],
            [row["Sleep Quality"], row["Sleep Quality"]],
            label="Sleep" if index == 0 else "",
            color="blue",
        )

    plt.title("Sleep Quality Over Time")
    plt.xlabel("Time")
    plt.ylabel("Sleep Quality")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
