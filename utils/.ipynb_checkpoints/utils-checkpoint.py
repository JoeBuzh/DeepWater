# -*- encoding: utf-8 -*-
'''
@Filename    : utils.py
@Datetime    : 2020/08/19 15:21:05
@Author      : Joe-Bu
@version     : 1.0
'''

import os
import sys

import numpy as np
import pandas as pd


def read_data(data_dir: str, filename: str) -> pd.DataFrame:
    """ 
    读取数据 & 修改列名 & 解析时间
    """
    filepath = os.path.join(data_dir, filename)
    assert os.path.exists(filepath)

    data = pd.read_excel(filepath, skiprows=1)
    data.drop(axis=0, index=0, inplace=True)
    data.rename(
        columns={
            '监测时间': 'Time', 
            '水温(℃)': 'Temp', 
            'pH(无量纲)': 'pH', 
            '溶解氧(mg/L)': 'DO', 
            '电导率(μS/cm)': 'Elecon', 
            '浊度(NTU)': 'Turbidity', 
            '高锰酸盐指数(mg/L)': 'CODMn', 
            '氨氮(mg/L)': 'NH3N', 
            '总磷(mg/L)': 'TP', 
            '总氮(mg/L)': 'TN'},
        inplace=True)
    data['Time'] = data['Time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))

    return data


def format_data(x):
    """
    数据预处理. 
    无效数据 & 负值 -> np.nan
    """ 
    # assert isinstance(x, str)
    if isinstance(x, float):
        return x
    assert isinstance(x, str)

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


def save_file(data, name=None):
    """
    输出文件保存
    """
    data.to_excel(name, index=False, 
        header=[
            '时间', '温度', 'pH', '溶解氧', '电导率', '浊度', 
            '高锰酸盐指数', '氨氮', '总磷', '总氮'
        ],
        encoding='utf-8')