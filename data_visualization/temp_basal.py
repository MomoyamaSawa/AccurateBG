import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取temp_basal数据
    timestamps_temp_basal_begin = []
    timestamps_temp_basal_end = []
    temp_basal_values = []

    for event in root.find("temp_basal"):
        timestamps_temp_basal_begin.append(
            datetime.strptime(event.get("ts_begin"), "%d-%m-%Y %H:%M:%S")
        )
        timestamps_temp_basal_end.append(
            datetime.strptime(event.get("ts_end"), "%d-%m-%Y %H:%M:%S")
        )
        temp_basal_values.append(float(event.get("value")))

    # 创建DataFrame
    df_temp_basal = pd.DataFrame(
        {
            "Timestamp Begin": timestamps_temp_basal_begin,
            "Timestamp End": timestamps_temp_basal_end,
            "Temp Basal Rate": temp_basal_values,
        }
    )

    # 可视化temp_basal
    plt.figure(figsize=(12, 6))

    for index, row in df_temp_basal.iterrows():
        plt.hlines(
            row["Temp Basal Rate"],
            xmin=row["Timestamp Begin"],
            xmax=row["Timestamp End"],
            colors="green",
            label="Temp Basal Rate" if index == 0 else "",
        )

    plt.title("Temp Basal Rate Over Time")
    plt.xlabel("Time")
    plt.ylabel("Temp Basal Rate (U/hr)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
