import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取illness数据
    timestamps_illness_begin = []
    illness_descriptions = []

    for event in root.find("illness"):
        timestamps_illness_begin.append(
            datetime.strptime(event.get("ts_begin"), "%d-%m-%Y %H:%M:%S")
        )
        illness_descriptions.append(event.get("description").strip())

    # 创建DataFrame
    df_illness = pd.DataFrame(
        {"Illness Begin": timestamps_illness_begin, "Description": illness_descriptions}
    )

    # 可视化illness数据
    plt.figure(figsize=(12, 6))

    plt.scatter(
        df_illness["Illness Begin"], [1] * len(df_illness), color="red", label="Illness"
    )
    for i, txt in enumerate(df_illness["Description"]):
        plt.annotate(
            txt,
            (df_illness["Illness Begin"][i], 1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.title("Illness Events Over Time")
    plt.xlabel("Time")
    plt.yticks([])
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
