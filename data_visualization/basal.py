import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取数据
    timestamps_basal = []
    basal_values = []

    for event in root.find("basal"):
        timestamps_basal.append(datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S"))
        basal_values.append(float(event.get("value")))

    # 创建DataFrame
    df_basal = pd.DataFrame({"Timestamp": timestamps_basal, "Basal Rate": basal_values})

    # 补全basal数据，确保每段时间有两个点以便绘制连续的直线
    basal_lines = []
    for i in range(len(df_basal) - 1):
        start_time = df_basal.loc[i, "Timestamp"]
        end_time = df_basal.loc[i + 1, "Timestamp"]
        basal_rate = df_basal.loc[i, "Basal Rate"]
        basal_lines.append((start_time, basal_rate))
        basal_lines.append((end_time, basal_rate))
    # 最后一个basal值延续到数据的最后时间
    basal_lines.append((df_basal["Timestamp"].iloc[-1], df_basal["Basal Rate"].iloc[-1]))
    basal_lines.append(
        (
            df_basal["Timestamp"].iloc[-1] + timedelta(hours=1),
            df_basal["Basal Rate"].iloc[-1],
        )
    )

    # 提取绘制basal线段的数据点
    basal_timestamps = [point[0] for point in basal_lines]
    basal_rates = [point[1] for point in basal_lines]

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(
        basal_timestamps,
        basal_rates,
        color="blue",
        label="Basal Rate",
        drawstyle="steps-post",
    )
    plt.title("Basal Rate Over Time")
    plt.xlabel("Time")
    plt.ylabel("Basal Rate (U/hr)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
