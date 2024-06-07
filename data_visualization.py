import xml.etree.ElementTree as ET
from enum import Enum


# 读取文件内容
file_path = "OhioT1DM/2018/test/588-ws-testing.xml"  # 替换为你的文件路径

with open(file_path, "r", encoding="utf-8") as file:
    xml_data = file.read()

# 将XML字符串解析为ElementTree对象
root = ET.fromstring(xml_data)


class DataType(Enum):
    """
    各种可视化数据图表的枚举
    """

    GLUCOSE_LEVEL_2_FINGER_STICK = "glucose_level-finger_stick"
    BASAL = "basal"
    TEMP_BASAL = "temp_basal"
    BOLUS = "bolus"
    MEAL = "meal"
    BASIS_GSR = "basis_gsr"
    BASIS_SKIN_TEMPERATURE = "basis_skin_temperature"
    BASIS_SLEEP = "basis_sleep"
    ACCELERATION = "acceleration"
    SLEEP = "sleep"
    WORK = "work"
    ILLNESS = "illness"
    EXERCISE = "exercise"
    BASIS_STEPS = "basis_steps"
    BASIS_AIR_TEMPERATURE = "basis_air_temperature"
    BASIS_HEART_RATE = "basis_heart_rate"
    HYPO_EVENT = "hypo_event"
    STRESSORS = "stressors"


datatype = DataType.GLUCOSE_LEVEL_2_FINGER_STICK

if datatype == DataType.GLUCOSE_LEVEL_2_FINGER_STICK:
    from data_visualization.glucose_level_2_finger_stick import show
elif datatype == DataType.BASAL:
    from data_visualization.basal import show
elif datatype == DataType.TEMP_BASAL:
    from data_visualization.temp_basal import show
elif datatype == DataType.BOLUS:
    from data_visualization.bolus import show
elif datatype == DataType.MEAL:
    from data_visualization.meal import show
elif datatype == DataType.BASIS_GSR:
    from data_visualization.basis_gsr import show
elif datatype == DataType.BASIS_SKIN_TEMPERATURE:
    from data_visualization.basis_skin_temperature import show
elif datatype == DataType.BASIS_SLEEP:
    from data_visualization.basis_sleep import show
elif datatype == DataType.ACCELERATION:
    from data_visualization.acceleration import show
elif datatype == DataType.SLEEP:
    from data_visualization.sleep import show
elif datatype == DataType.WORK:
    from data_visualization.work import show
elif datatype == DataType.ILLNESS:
    from data_visualization.illness import show
elif datatype == DataType.EXERCISE:
    from data_visualization.exercise import show
elif datatype == DataType.BASIS_STEPS:
    from data_visualization.basis_steps import show
elif datatype == DataType.BASIS_AIR_TEMPERATURE:
    from data_visualization.basis_air_temperature import show
elif datatype == DataType.BASIS_HEART_RATE:
    from data_visualization.basis_heart_rate import show
elif datatype == DataType.HYPO_EVENT:
    from data_visualization.hypo_event import show
elif datatype == DataType.STRESSORS:
    from data_visualization.stressors import show


show(root)
