from . import model_1
from . import model_2
import pandas as pd

def estimate(data: pd.DataFrame, str_x, str_y):
    model1 = model_1.model_1()
    result1 = model1.estimate(data, str_x, str_y)
    list1 = [data['code'].iloc[0], data['stock name'].iloc[0], model1.name(), result1[0], result1[1]]
    model2 = model_2.model_2()
    result2 = model2.estimate(data, str_x, str_y)
    list2 = [data['code'].iloc[0], data['stock name'].iloc[0], model2.name(), result2[0], result1[1]]
    return (pd.DataFrame([list1, list2], columns=['code', 'stock name', 'model name', 'pipeline', 'MSR']))