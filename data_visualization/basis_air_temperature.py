import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取basis_air_temperature数据
    timestamps_air_temp = []
    air_temp_values = []

    for event in root.find("basis_air_temperature"):
        timestamps_air_temp.append(
            datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S")
        )
        air_temp_values.append(float(event.get("value")))

    # 创建DataFrame
    df_air_temp = pd.DataFrame(
        {"Timestamp": timestamps_air_temp, "Air Temperature": air_temp_values}
    )

    # 可视化basis_air_temperature数据
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_air_temp["Timestamp"],
        df_air_temp["Air Temperature"],
        color="green",
        label="Air Temperature",
    )
    plt.title("Air Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Air Temperature (°F)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
