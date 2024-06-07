import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def show(root: ET.Element):

    # 提取meal数据
    timestamps_meal = []
    meal_types = []
    carbs_values = []

    for event in root.find("meal"):
        timestamps_meal.append(datetime.strptime(event.get("ts"), "%d-%m-%Y %H:%M:%S"))
        meal_types.append(event.get("type"))
        carbs_values.append(float(event.get("carbs")))

    # 创建DataFrame
    df_meal = pd.DataFrame(
        {"Timestamp": timestamps_meal, "Meal Type": meal_types, "Carbs": carbs_values}
    )

    # 可视化meal数据
    plt.figure(figsize=(12, 6))

    for meal_type in df_meal["Meal Type"].unique():
        subset = df_meal[df_meal["Meal Type"] == meal_type]
        if meal_type == "Breakfast":
            plt.scatter(
                subset["Timestamp"],
                subset["Carbs"],
                color="blue",
                label=meal_type,
                marker="o",
            )
        elif meal_type == "Lunch":
            plt.scatter(
                subset["Timestamp"],
                subset["Carbs"],
                color="green",
                label=meal_type,
                marker="s",
            )
        elif meal_type == "Dinner":
            plt.scatter(
                subset["Timestamp"],
                subset["Carbs"],
                color="red",
                label=meal_type,
                marker="^",
            )
        elif meal_type == "Snack":
            plt.scatter(
                subset["Timestamp"],
                subset["Carbs"],
                color="purple",
                label=meal_type,
                marker="x",
            )

    # 设置图例和其他属性
    plt.title("Carbohydrate Intake Over Time")
    plt.xlabel("Time")
    plt.ylabel("Carbohydrates (g)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
