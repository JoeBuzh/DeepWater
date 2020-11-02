# -*- encoding: utf-8 -*-
'''
@Filename    : fbprophet.py
@Datetime    : 2020/08/31 14:44:46
@Author      : Joe-Bu
@version     : 1.0
'''

import os
import sys

import numpy as np
import pandas as pd

import fbprophet
from fbprophet import Prophet


class ProphetModel(object):
    """
    Fbprophet Model Class.
    """
    
    def __init__(self, periods: int, freq: int, index: str):
        self.periods = periods
        self.freq = freq
        self.index = index
    
    @staticmethod
    def _define(growth: str, last_days: int, n_chg_points: int):
        '''
        Model Defination.
        '''
        if last_days <= 15:
            raise MinHistError('Fit History Length Less Than 15 Days.')
        
        return Prophet(growth=growth)
    
    @staticmethod
    def _train(model, data: pd.DataFrame, index: str) -> tuple:
        '''
        Train Method.
        '''
        y_mean = data['y'].mean()
        y_std = data['y'].std()
        data['y'] = (data['y']-y_mean) / y_std
        
        model.fit(data)
        
        return model, y_mean, y_std
    
    @staticmethod
    def _predict(model, future: pd.DataFrame, y_mean, y_std) -> pd.DataFrame:
        '''
        Predict Method.
        '''
        prediction = model.predict(future)
        forecast = prediction[['yhat_lower', 'yhat_upper', 'yhat']].apply(lambda x: x*y_std+y_mean)
        forecast['ds'] = prediction['ds']
        
        return forecast
    
    def _make_timestamps(self, model: fbprophet) -> pd.DataFrame:
        '''
        Define Forecast Timestamps.
        '''
        return model.make_future_dataframe(
            periods=self.periods,
            freq='{}h'.format(self.freq)
        )
    
    
class MinHistError(BaseException):
    """
    Minimal History Length Error.
    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
        

def test():
    model = ProphetModel(periods=3, freq=5, index='a')
    print(model)
    
    
if __name__ == "__main__":
    test()