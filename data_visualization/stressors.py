import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取stressors数据
    timestamps_stressors = []

    for event in root.find("stressors"):
        timestamps_stressors.append(
            datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S")
        )

    # 创建DataFrame
    df_stressors = pd.DataFrame({"Timestamp": timestamps_stressors})

    # 可视化stressors数据
    plt.figure(figsize=(12, 6))

    plt.scatter(
        df_stressors["Timestamp"],
        [1] * len(df_stressors),
        color="orange",
        label="Stress Event",
    )
    for i, txt in enumerate(df_stressors["Timestamp"]):
        plt.annotate(
            txt.strftime("%Y-%m-%d %H:%M"),
            (df_stressors["Timestamp"][i], 1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.title("Stress Events Over Time")
    plt.xlabel("Time")
    plt.yticks([])
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
