# -*- encoding: utf-8 -*-
'''
@Filename    : main.py
@Datetime    : 2020/08/19 16:12:29
@Author      : Joe-Bu
@version     : 1.0
'''

import os
import sys
sys.path.append('../../')
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from src.DataSimulation.fileport import read_data, save_data
from src.DataSimulation.utils import format_data, transform_data
from src.DataSimulation.utils import wipe_anomaly, expand_time
from src.DataSimulation.model import get_dataset, train_split_std
from src.DataSimulation.model import model_load, model_save, poly_features
from src.DataSimulation.model import model_train, model_predict, model_evaluate

from lib.machineL.NonLinearModel import xgb
from lib.machineL.NonLinearModel import gbrt
from lib.machineL.NonLinearModel import rf


def load_data(data_dir: str, filename: str) -> pd.DataFrame:
    '''
    读取并解析数据内容
    '''
    # 读取
    data = read_data(data_dir, filename)
    # 解析
    data = transform_data(data)
    return data.replace([-999.0, 999.0], [np.nan, np.nan])
    

def get_4h_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
    获取4小时频率数据
    '''
    return pd.concat([data[data['Time'].dt.hour==i] for i in range(0, 21 ,4)]).sort_index()


def load_cfg() -> tuple:
    '''
    获取配置
    '''
    from cfg import paths, files, model_params

    return paths, files, model_params


def train(model_name: str, index: str, train_src: pd.DataFrame, paths: dict):
    """ 
    模型训练过程 
    """
    if model_name == 'RF':
        model_in = rf(
            n_estimators=300, max_features=0.7, oob_score=True)
    elif model_name == 'GBRT':
        model_in = gbrt(
            n_estimators=300, learning_rate=1, subsample=0.7, max_features=0.6)
    elif model_name == 'XGB':
        model_in = xgb(n_estimators=300, subsample=0.7)
    else:
        print('Bad Model Selection')
        sys.exit()
    dataset = get_dataset(train_src, columns=['Temp', 'pH', 'DO', 'Elecon', 'Turbidity', index])
    print(dataset.sample(3))
    print(dataset.shape)
    X_train, X_test, y_train, y_test, std_x, std_y = train_split_std(dataset, index=index, method='STD')
    poly = poly_features(degree=3)

    model_out = model_train(
        model_in, 
        poly.fit_transform(X_train), 
        y_train)
    train_err = model_evaluate(
        std_y.inverse_transform(y_train), 
        std_y.inverse_transform(model_predict(model_out, poly.fit_transform(X_train))))
    test_err = model_evaluate(
        std_y.inverse_transform(y_test), 
        std_y.inverse_transform(model_predict(model_out, poly.fit_transform(X_test))))
    print(train_err, test_err)
    model_save(
        model_out, 
        os.path.join(paths['model_dir'], '{0}_{1}.pkl'.format(model_name, index))
    )


def predict(model_name: str, index_name: str, pred_data: pd.DataFrame, paths: dict):
    """
    模型预测过程
    """
    data_qc = wipe_anomaly(pred_data)
    model = model_load(os.path.join(paths['model_dir'], '{0}_{1}.pkl'.format(model_name, index_name)))
    data = data_qc[['Temp', 'pH', 'DO', 'Elecon', 'Turbidity', index_name]]
    test_std_x = StandardScaler().fit(data[['Temp','pH','DO','Elecon','Turbidity']])
    test_std_y = StandardScaler().fit(data[[index_name]])
    poly = poly_features(degree=3)

    # data['{}_P'.format(index_name)] = test_std_y.inverse_transform(
    data['{}_P'.format(index_name)] = test_std_y.inverse_transform(
        model.predict(
            poly.fit_transform(
                test_std_x.transform(
                    data[['Temp','pH','DO','Elecon','Turbidity']]
                )
            )
        )
    ).reshape(-1, 1)
    print(pred_data.shape)
    print(data.shape)
    pred_data['{}_P'.format(index_name)] = data['{}_P'.format(index_name)]

    # return data


def main():
    ''' 主流程 '''
    # --> cfg
    paths, filenames, model_params = load_cfg()
    # --> raw data
    data_dawj = load_data(paths['rawdata_dir'], filenames['dawj'])
    data_wuys = load_data(paths['rawdata_dir'], filenames['wuys'])
    data_yanjd = load_data(paths['rawdata_dir'], filenames['yanjd'])
    # --> test data
    test_dawj = load_data(paths['testdata_dir'], filenames['dawj_test'])
    # test_wuys = load_data(paths['testdata_dir'], filenames['wuys_test'])
    # test_yanjd = load_data(paths['testdata_dir'], filenames['yanjd_test'])
    test_lncs = load_data(paths['testdata_dir'], filenames['lncs_test'])

    raw_data = data_yanjd
    test_data = test_lncs

    # --> extract
    test_4h = get_4h_data(test_data)
    print(test_4h.describe(percentiles=[.01,.02,.03,.05,.9,.95,.97,.99]))
    # --> expand
    test_4h_expand = expand_time(test_4h)
    print(test_4h_expand.describe(percentiles=[.01,.02,.03,.05,.9,.95,.97,.99]))

    # sys.exit()
    # --> save
    save_data(
      test_4h_expand, 
      os.path.join(paths['output_dir'], '洛宁长水4h验证原始数据(1)-线性扩展.xls')
    )

    # --> concat
    raw_4h_list = []
    for i, data in enumerate([data_dawj, data_wuys, data_yanjd]):
        data_4h = get_4h_data(data)
        print(data_4h)
        raw_4h_list.append(data_4h)

    # --> modeling
    index = model_params['index']
    model = model_params['model']
    modes = model_params['modes']

    if modes == 'train':
        # train(model, index, raw_data, paths)
        train(model, index, pd.concat(raw_4h_list), paths)

    elif modes == 'predict':
        predict(model, index, test_4h_expand, paths)

    elif modes == 'predicts':
        data_list = []
        for i, index in enumerate(model_params['indexs']):
            predict(model, index, test_4h_expand, paths)

        print(test_4h_expand.sample(5))  
        save_data(
            test_4h_expand[
                ['Time', 'Temp', 'pH', 'DO', 'Elecon', 
                 'Turbidity', 'CODMn_P', 'NH3N_P', 'TP_P', 'TN_P']],
            os.path.join(paths['output_dir'], '洛宁长水4h验证原始数据(1)-机器学习扩展.xls')
        )

    else:
        print('Wrong Modes.')
        sys.exit()


if __name__ == "__main__":
    main()