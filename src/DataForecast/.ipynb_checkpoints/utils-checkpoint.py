# -*- encoding: utf-8 -*-
'''
@Filename    : utils.py
@Datetime    : 2020/08/31 16:22:42
@Author      : Joe-Bu
@version     : 1.0
'''

"""
ARIMA 预测模型辅助工具
"""

import os
import sys

import numpy as np
import pandas as pd


def read_data(absfile: str) -> pd.DataFrame:
    """ 
    Read Raw Data.
    """
    assert os.path.exists(absfile)

    return pd.read_excel(absfile,
        names=[
            'Time', 'Temp', 'pH', 'DO', 'Elecon', 'Turbidity', 'CODMn', 'NH3N', 'TP', 'TN'
        ])


def save_data(data: pd.DataFrame, absfile: str):
    """ 
    Save Forecast Data.
    """
    data.to_excel(
        absfile,
        encoding='utf-8'
    )


def read_forecast(absfile: str) -> pd.DataFrame:
    """
    Read Forecast Data.
    """
    return pd.read_excel(absfile, encoding='utf-8', index_col=0)


def checkdir(path: str) -> str:
    """
    Check Dirname.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    assert os.path.exists(path)

    return path