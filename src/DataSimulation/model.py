# -*- encoding: utf-8 -*-
'''
@Filename    : model.py
@Datetime    : 2020/08/19 17:20:42
@Author      : Joe-Bu
@version     : 1.0
'''
import os
import sys
sys.path.append('../../')

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.SimulateUtils import wipe_anomaly


def get_cols_data(data: pd.DataFrame, columns: list, lower: float, upper: float) -> pd.DataFrame:
    """
    获取训练数据集
        data: 提取数据集
        columns:        feature_col + y_col
        columns[:-1]:   feature_cols
        cloumns[-1]:    y_col
    """
    print(columns)
    
    return wipe_anomaly(data[columns], lower, upper).dropna()


def model_load(model_path: str):
    """ 模型加载 """
    assert os.path.exists(model_path)

    return joblib.load(model_path)


def model_save(model, model_path: dir):
    """ 模型保存 """
    joblib.dump(model, model_path)

    assert os.path.exists(model_path)


def model_train(model, X, y):
    """ 模型训练 """
    print(model)
    model.fit(X, y)

    return model


def model_predict(model, X):
    """ 模型预测 """
    return model.predict(X)


def model_evaluate(y, y_pred):
    """ 模型评估 """
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    return mean_squared_error(y, y_pred)


def train_split_std(data: pd.DataFrame, index: str, method='MinMax') -> tuple:
    """ 📦
    train_test_split & 归一化 & 标准化
    """
    # print(data.drop([index], axis=1).sample(5))
    # print(data[[index]].sample(5))
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([index], axis=1).sort_index(),
        data[[index]].sort_index(),
        test_size=.2,
        random_state=2048
    )
    # 先对 X_train set 进行预处理，所得指标用于 valid set & test set
    if method == 'MinMax':
        # MinMax
        minmax_x = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
        minmax_y = MinMaxScaler(feature_range=(0, 1)).fit(y_train)

        X_train[X_train.columns] = minmax_x.transform(X_train)
        X_test[X_test.columns] = minmax_x.transform(X_test)

        y_train[y_train.columns] = minmax_y.transform(y_train)
        y_test[y_test.columns] = minmax_y.transform(y_test)

        return X_train, X_test, y_train, y_test, minmax_x, minmax_y

    elif method == 'STD':
        # Standard
        std_x = StandardScaler().fit(X_train)
        std_y = StandardScaler().fit(y_train)

        X_train[X_train.columns] = std_x.transform(X_train)
        X_test[X_test.columns] = std_x.transform(X_test)

        y_train[y_train.columns] = std_y.transform(y_train)
        y_test[y_test.columns] = std_y.transform(y_test)

        return X_train, X_test, y_train, y_test, std_x, std_y

    else:
        # Raw 
        return X_train, X_test, y_train, y_test, None, None


def poly_features(degree: int):
    """ 高阶交叉特征 """
    from sklearn.preprocessing import PolynomialFeatures

    return PolynomialFeatures(degree)
