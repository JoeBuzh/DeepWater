# -*- encoding: utf-8 -*-
'''
@Filename    : utils.py
@Datetime    : 2020/08/31 16:18:07
@Author      : Joe-Bu
@version     : 1.0
'''

""" 评价辅助工具 """

import os

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def calc_accuracy():
    pass
        
        
def calc_mae(y_true, y_pred) -> float:
    """
    Calculate Mean Absolute Error.
        y_true: Ture Observation.
        y_pred: Forecast Value.
    """
    return mean_absolute_error(y_true, y_pred)
    
    
def calc_mse(y_true, y_pred) -> float:
    """
    Calculate Mean Squared Error.
        y_true: Ture Observation.
        y_pred: Forecast Value.
    """
    return mean_squared_error(y_true, y_pred)


def calc_nash(y_true, y_pred) -> float:
    """
    Calculate Nash Correlation Index.
        y_true: Ture Observation.
        y_pred: Forecast Value.
    """
    return r2_score(y_true, y_pred)