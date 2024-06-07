import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取basis_gsr数据
    timestamps_gsr = []
    gsr_values = []

    for event in root.find("basis_gsr"):
        timestamps_gsr.append(datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S"))
        gsr_values.append(float(event.get("value")))

    # 创建DataFrame
    df_gsr = pd.DataFrame({"Timestamp": timestamps_gsr, "GSR": gsr_values})

    # 可视化basis_gsr数据
    plt.figure(figsize=(12, 6))
    plt.plot(df_gsr["Timestamp"], df_gsr["GSR"], color="blue", label="GSR")
    plt.title("Galvanic Skin Response (GSR) Over Time")
    plt.xlabel("Time")
    plt.ylabel("GSR (μS)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
