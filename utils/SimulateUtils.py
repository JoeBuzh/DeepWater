# -*- encoding: utf-8 -*-
'''
@Filename    : SimulateUtils.py
@Datetime    : 2020/09/24 10:12:53
@Author      : Joe-Bu
@version     : 1.0
'''

import numpy as np
import pandas as pd


def format_data(x):
    """ 
    *** 针对原始文本数据 ***
      1. 处理字符问题 Data Formatter. 
      2. 无效数据、负值变为np.nan 
    """
    # assert isinstance(x, str)
    if isinstance(x, float):
        return x
        
    elif  isinstance(x, str):
        if x[-2:] in ['HD', 'LW', 'LR', 'LS']:
            num = float(x[:-2])
        elif x[-1] in ['N', 'T', 'S']:
            num = float(x[:-1])
        elif x[-1] in ['L', 'P', 'D', 'F', 'B', 'Z', 'M']:
            num = np.nan
        else:
            num = float(x)

        if num > 0.0:
            return num
        else:
            return np.nan


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """ 
    *** 针对文本数据 ***
      1. 格式化字符数据，对不同后缀的数据进行解析处理 
    """
    for i, col in enumerate(data.columns[1:]):
        data[col] = data[col].apply(format_data)

    return data


def wipe_anomaly(data: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """
    根据每个 col 的分布，其分布 lower以下、upper以上 数据视为异常，重置为 np.nan
    """
    tmp = data[data.columns]

    for _, col in enumerate(tmp.columns[1:]):
        tmp.loc[tmp[col]<tmp[col].quantile(lower), col] = np.nan
        tmp.loc[tmp[col]>tmp[col].quantile(upper), col] = np.nan
    
    # return tmp.interpolate(method='linear').fillna(method='bfill')
    return tmp


def expand_time(data: pd.DataFrame):
    """
    扩展时间纬度信息
    """
    tmp = pd.DataFrame(columns=data.columns)
    tmp['Time'] = pd.date_range(
        data['Time'].values[0],
        data['Time'].values[-1],
        freq='4H'
    )
    tmp = tmp.merge(wipe_anomaly(data), how='outer').groupby('Time').max()

    return tmp.interpolate(method='linear').fillna(method='bfill').reset_index('Time')
