# -*- encoding: utf-8 -*-
'''
@Filename    : ARIMA.py
@Datetime    : 2020/08/31 14:44:46
@Author      : Joe-Bu
@version     : 1.0
'''

import os
import sys
import numpy as np
import pandas as pd

from scipy import stats

import statsmodels.api as sm                            # 一阶自相关性检验
from statsmodels.tsa.arima_model import ARIMA           # ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose # 序列分解
from statsmodels.tsa.stattools import adfuller as ADF   # 平稳性检验
from statsmodels.graphics.tsaplots import plot_acf      # ACF
from statsmodels.graphics.tsaplots import plot_pacf     # PACF
from statsmodels.graphics.api import qqplot             # 检验残差服从正太分布
from statsmodels.stats.diagnostic import acorr_ljungbox # 白噪声检验


class ArimaModel:
    """
    ARIMA滑动平均差分自回归模型.
    """
    def __init__(self, data: pd.DataFrame, p: int, d: int, q: int):
        '''
        Init.
        '''
        self.data = data
        self.p = p
        self.d = d
        self.q = q
        # self.model = self.__define()

    def _define(self):
        """
        Model Define.
        """
        self.model = ARIMA(self.data, order=(self.p, self.d, self.q))

    def show_params(self):
        """
        Display p, d, q
        """
        print('ARIMA Model: p={0} d={1} q={2}'.format(self.p, self.d, self.q))

    def _train(self):
        '''
        Train.
        '''
        return self.model.fit(disp=-1, method='css', start_ar_lags=13)

    def _predict(self, start, end):
        '''
        Predict
        '''
        self.model.predict(start=start, end=end, typ='levels')

    def _forecast(self, pred_len: int):
        '''
        Forecast.
        '''
        return self.model.forecast(pred_len)