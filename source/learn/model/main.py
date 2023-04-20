from . import model_1
from . import model_2
from . import model_3
import pandas as pd

def estimate(data: pd.DataFrame, str_x, str_y):
    model1 = model_1.model_1(
        data['code'].iloc[0], data['stock name'].iloc[0],
        data[str_x], data[str_y]
    )
    #result1 = model1.estimate(data, str_x, str_y)
    #list1 = [data['code'].iloc[0], data['stock name'].iloc[0], model1.name(), result1[0], data[str_x].iloc[-1], result1[1], result1[2]]
    model2 = model_2.model_2(
        data['code'].iloc[0], data['stock name'].iloc[0],
        data[str_x], data[str_y]
    )
    #result2 = model2.estimate(data, str_x, str_y)
    #list2 = [data['code'].iloc[0], data['stock name'].iloc[0], model2.name(), result2[0], data[str_x].iloc[-1], result2[1], result2[2]]
    model3 = model_3.model_3(
        data['code'].iloc[0], data['stock name'].iloc[0],
        data[str_x], data.filter([str_y]).values
    )
    #result3 = model3.estimate(data, str_x, str_y)
    #list3 = [data['code'].iloc[0], data['stock name'].iloc[0], model3.name(), result3[0], data[str_x].iloc[-1], result3[1], result3[2]]
    return (pd.Series([model1, model2, model3]))
