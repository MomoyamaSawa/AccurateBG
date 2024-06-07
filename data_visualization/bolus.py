import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取bolus数据
    timestamps_bolus_begin = []
    timestamps_bolus_end = []
    bolus_values = []
    bolus_types = []

    for event in root.find("bolus"):
        timestamps_bolus_begin.append(
            datetime.strptime(event.get("ts_begin"), "%d-%m-%Y %H:%M:%S")
        )
        timestamps_bolus_end.append(
            datetime.strptime(event.get("ts_end"), "%d-%m-%Y %H:%M:%S")
        )
        bolus_values.append(float(event.get("dose")))
        bolus_types.append(event.get("type"))

    # 创建DataFrame
    df_bolus = pd.DataFrame(
        {
            "Timestamp Begin": timestamps_bolus_begin,
            "Timestamp End": timestamps_bolus_end,
            "Bolus Value": bolus_values,
            "Bolus Type": bolus_types,
        }
    )

    # 可视化bolus
    plt.figure(figsize=(12, 6))

    for index, row in df_bolus.iterrows():
        if row["Bolus Type"] == "normal":
            plt.scatter(
                row["Timestamp Begin"],
                row["Bolus Value"],
                color="blue",
                label="Normal Bolus" if index == 0 else "",
            )
        elif row["Bolus Type"] == "normal dual":
            plt.scatter(
                row["Timestamp Begin"],
                row["Bolus Value"],
                color="purple",
                label="Normal Dual Bolus" if index == 0 else "",
            )
        elif row["Bolus Type"] == "square dual":
            plt.hlines(
                row["Bolus Value"],
                xmin=row["Timestamp Begin"],
                xmax=row["Timestamp End"],
                colors="orange",
                label="Square Dual Bolus" if index == 0 else "",
            )

    plt.title("Bolus Over Time")
    plt.xlabel("Time")
    plt.ylabel("Bolus Value (U)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
