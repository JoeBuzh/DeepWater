# -*- encoding: utf-8 -*-
'''
@Filename    : model.py
@Datetime    : 2020/08/31 16:30:10
@Author      : Joe-Bu
@version     : 1.0
'''

"""
ARMIA 模型参数选取
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime,timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm                            # 一阶自相关性检验
from statsmodels.tsa.arima_model import ARIMA           # ARIMA & ARMA
from statsmodels.tsa.seasonal import seasonal_decompose # 序列分解
from statsmodels.tsa.stattools import adfuller as ADF   # 平稳性检验
from statsmodels.graphics.tsaplots import plot_acf      # ACF
from statsmodels.graphics.tsaplots import plot_pacf     # PACF
from statsmodels.graphics.api import qqplot             # 检验残差服从正太分布
from statsmodels.stats.diagnostic import acorr_ljungbox # 白噪声检验


def decompose(data: pd.DataFrame) -> tuple:
    """
    STL时间序列分解 
    """
    decomposition = seasonal_decompose(data)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return trend, seasonal, residual


def decompose_plot(group: list):
    """
    trend, seasonal, residual = decompose(raw)
    group = [
        (raw, 'Original'), (trend, 'Trend'), 
        (seasonal, 'Seasonal'), (residual, 'Residual')]
    """
    plt.figure(figsize=(16, 12))
    for i, (data, name) in enumerate(group):
        plt.subplot(4, 1, i+1)
        plt.plot(data, label=name)
        plt.legend(loc='best')
    plt.show()


def adf_diff(data: pd.DataFrame, plot: bool=False) -> int:
    """
    ADF检验 -> d 
    """
    # diff & fillna
    data_diff1 = data.diff(1).fillna(0.0)
    data_diff2 = data.diff(1).diff(1).fillna(0.0)
    # ADF
    data_adf = ADF(data)
    data_diff1_adf = ADF(data_diff1)
    data_diff2_adf = ADF(data_diff2)
    # get p
    p = 0
    for i, adf in enumerate([data_adf, data_diff1_adf, data_diff2_adf]):
        t_val, p_val, _, _, ts, _ = adf
        if t_val < min(ts.values()):
            p = i
            print('p={}\nadf={}'.format(i, adf))
            break
        else:
            p += i
    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(data, label='Original', color='blue')
        plt.plot(data_diff1, label='Diff1', color='red')
        plt.plot(data_diff2, label='Diff2', color='green')
        plt.legend(loc='best')
        plt.title("{}".format(index))
        plt.show()

    return p


def adf_calc(data: pd.DataFrame) -> int:
    """
    ADF[1] vs. 0.05
    """
    p = 0
    if ADF(data)[1] < 0.05:
        print('Raw ADF: {}'.format(ADF(data)[1]))
        return p
    elif ADF(data.diff(1).dropna())[1] < 0.05:
        print('Diff1 ADF: {}'.format(ADF(data.diff(1).dropna())[1]))
        return p + 1
    elif ADF(data.diff(1).diff(1).dropna())[1] < 0.05:
        print('Diff2 ADF: {}'.format(ADF(data.diff(1).diff(1).dropna())[1]))
        return p + 2

    
def get_order(data: pd.DataFrame) -> tuple:
    """ 
    获取最佳 p & q 
    """
    res = sm.tsa.arma_order_select_ic(
        data,
        ic=['bic'],
        trend='nc',
        max_ar=5,
        max_ma=5
    )
    print(res.bic_min_order)

    return res.bic_min_order


def resid_check(resid) -> bool:
    """
    ARIMA Resid Check.
    """
    val = sm.stats.durbin_watson(resid.values)
    print('Resid Chcek: {}'.format(val))

    if abs(val-2.0) <= 0.05:
        return True
    else:
        return False
    