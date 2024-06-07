import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取basis_steps数据
    timestamps_steps = []
    steps_values = []

    for event in root.find("basis_steps"):
        timestamps_steps.append(datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S"))
        steps_values.append(int(event.get("value")))

    # 创建DataFrame
    df_steps = pd.DataFrame({"Timestamp": timestamps_steps, "Steps": steps_values})

    # 可视化basis_steps数据
    plt.figure(figsize=(12, 6))
    plt.plot(df_steps["Timestamp"], df_steps["Steps"], color="blue", label="Steps")
    plt.title("Steps Over Time")
    plt.xlabel("Time")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
