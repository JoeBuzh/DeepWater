# -*- encoding: utf-8 -*-
'''
@Filename    : NonLinearModel.py
@Datetime    : 2020/08/20 10:22:49
@Author      : Joe-Bu
@version     : 1.0
'''

""" 初始化对应算法后再倒入sklearn对应模块 """


def RF(n_estimators: int, max_features: float, oob_score: bool):
    """ RandomForestRegressor """
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=n_estimators, 
        max_features=max_features, 
        oob_score=oob_score) 


def ERT(n_estimators: int, oob_score: bool):
    """ Extremely Randomized Trees """
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=n_estimators, 
        oob_score=oob_score)


def ADB(max_depth: int, max_features: float, n_estimators: int, learning_rate: float):
    """ AdaBoost """
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor

    return AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth, max_features), 
        n_estimators=n_estimators, 
        learning_rate=learning_rate,
        loss='square')


def GBRT(n_estimators: int, learning_rate: float, subsample: float, max_features: float):
    """ GBRT """
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        subsample=subsample,
        max_features=max_features)


def XGB(n_estimators: int, subsample: float):
    """ XGBoost """
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators, 
        booster='gbtree', 
        subsample=subsample, 
        colsample_bytree=0.5)