import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取hypo_event数据
    timestamps_hypo = []

    for event in root.find("hypo_event"):
        timestamps_hypo.append(datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S"))

    # 创建DataFrame
    df_hypo = pd.DataFrame({"Timestamp": timestamps_hypo})

    # 可视化hypo_event数据
    plt.figure(figsize=(12, 6))

    plt.scatter(
        df_hypo["Timestamp"], [1] * len(df_hypo), color="red", label="Hypo Event"
    )
    for i, txt in enumerate(df_hypo["Timestamp"]):
        plt.annotate(
            txt.strftime("%Y-%m-%d %H:%M"),
            (df_hypo["Timestamp"][i], 1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.title("Hypo Events Over Time")
    plt.xlabel("Time")
    plt.yticks([])
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
