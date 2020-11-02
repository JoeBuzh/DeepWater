# -*- encoding: utf-8 -*-
'''
@Filename    : LinearModel.py
@Datetime    : 2020/08/20 10:20:10
@Author      : Joe-Bu
@version     : 1.0
'''

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

""" 线性模型 """

def lr():
    return LinearRegression(fit_intercept=True, normalize=True)


def lasso():
    return Lasso(alpha=0.8, fit_intercept=True, normalize=True)


def ridge():
    return Ridge(alpha=0.8, fit_intercept=True, normalize=True)


def elasticNet():
    return ElasticNet(alpha=0.8, fit_intercept=True, normalize=True)