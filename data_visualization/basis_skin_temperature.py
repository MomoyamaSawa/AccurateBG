import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取basis_skin_temperature数据
    timestamps_skin_temp = []
    skin_temp_values = []

    for event in root.find("basis_skin_temperature"):
        timestamps_skin_temp.append(
            datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S")
        )
        skin_temp_values.append(float(event.get("value")))

    # 创建DataFrame
    df_skin_temp = pd.DataFrame(
        {"Timestamp": timestamps_skin_temp, "Skin Temperature": skin_temp_values}
    )

    # 可视化basis_skin_temperature数据
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_skin_temp["Timestamp"],
        df_skin_temp["Skin Temperature"],
        color="red",
        label="Skin Temperature",
    )
    plt.title("Skin Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Skin Temperature (°F)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
