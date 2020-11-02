# -*- encoding: utf-8 -*-
'''
@Filename    : main.py
@Datetime    : 2020/08/19 16:12:29
@Author      : Joe-Bu
@version     : 1.0
'''

import os
import sys
from copy import deepcopy
sys.path.append('../../')
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import and_, between
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from src.DataSimulation.model import get_cols_data, train_split_std
from src.DataSimulation.model import model_load, model_save, poly_features
from src.DataSimulation.model import model_train, model_predict, model_evaluate

from config.simulate_settings import model_params, NORMAL_ITEMS, NORMAL_INDEX
from dao.orm.SQLite_dynamic import orm
from dao.orm.SQLite_dynamic import ObsDataRaw, ObsDataQcNonLinear
from lib.machineL.NonLinearModel import XGB, GBRT, RF
from utils.SimulateUtils import wipe_anomaly
from utils.ThreadUtils import WorkThread


def query_data(session, station_name: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    读取 & 解析数据.
    """
    time_format = "%Y-%m-%d %H:00:00"
    resp = session.query(
        ObsDataRaw.time,
        ObsDataRaw.watertemp,
        ObsDataRaw.pH,
        ObsDataRaw.DO,
        ObsDataRaw.conductivity,
        ObsDataRaw.turbidity,
        ObsDataRaw.codmn,
        ObsDataRaw.nh3n,
        ObsDataRaw.tp,
        ObsDataRaw.tn) \
            .filter_by(name=station_name) \
            .filter(between(ObsDataRaw.time, start.strftime(time_format), end.strftime(time_format))) \
            .all()
    data = pd.DataFrame(resp)
    
    return data.replace([-999.0, 999.0], [np.nan, np.nan])


def insert_data(session, station_name: str, model_name,data: pd.DataFrame):
    """
    数据写入.
    """
    inserts = []
    for i, row in data.iterrows():
        ins = ObsDataQcNonLinear(
            time=row['time'],
            name=station_name,
            method=model_name,
            watertemp=row['watertemp'],
            pH=row['pH'],
            DO=row['DO'],
            conductivity=row['conductivity'],
            turbidity=row['turbidity'],
            codmn=row['codmn_hat'],
            nh3n=row['nh3n_hat'],
            tp=row['tp_hat'],
            tn=row['tn_hat'])
        inserts.append(ins)
        
    session.add_all(inserts)
    session.flush()


def train(model_name: str, index_name: str, train_data: pd.DataFrame) -> bool:
    """ 
    模型训练过程.
    """ 
    if model_name == 'RF':
        model_in = RF(n_estimators=300, max_features=0.7, oob_score=True)
    elif model_name == 'GBRT':
        model_in = GBRT(n_estimators=300, learning_rate=1, subsample=0.7, max_features=0.6)
    elif model_name == 'XGB':
        model_in = XGB(n_estimators=300, subsample=0.7)
    else:
        print('Bad Model Selection')
        sys.exit()

    dataset = get_cols_data(train_data, columns=NORMAL_ITEMS+[index_name], lower=.05, upper=.95)
    X_train, X_test, y_train, y_test, std_x, std_y = train_split_std(dataset, index=index_name, method='STD')
    poly = poly_features(degree=3)
    print("{0}:: Train Size: {1}".format(index_name, X_train.shape))

    # sys.exit()

    model_out = model_train(model_in, poly.fit_transform(X_train), y_train)
    train_err = model_evaluate(
        std_y.inverse_transform(y_train), 
        std_y.inverse_transform(model_predict(model_out, poly.fit_transform(X_train))))
    test_err = model_evaluate(
        std_y.inverse_transform(y_test), 
        std_y.inverse_transform(model_predict(model_out, poly.fit_transform(X_test))))
    print("{0}:: Train Error:{1}\nTest Error:{2}".format(index_name, train_err, test_err))
    model_save(model_out, 
        os.path.join(model_params['savedir'], '{0}_{1}.pkl'.format(model_name, index_name)))

    if os.path.exists(os.path.join(model_params['savedir'], '{0}_{1}.pkl'.format(model_name, index_name))):
        return True
    else:
        return False


def predict(model_name: str, index_name: str, pred_data: pd.DataFrame):
    """
    模型预测过程.
    """
    temp_data = deepcopy(pred_data)
    data_qc = wipe_anomaly(data=temp_data, lower=.05, upper=.95)
    model = model_load(os.path.join(model_params['savedir'], '{0}_{1}.pkl'.format(model_name, index_name)))

    data = data_qc[NORMAL_ITEMS+[index_name]].interpolate(method='linear').fillna(method='bfill')
    test_std_x = StandardScaler().fit(data[NORMAL_ITEMS])
    test_std_y = StandardScaler().fit(data[[index_name]])
    poly = poly_features(degree=3)

    data['{}_hat'.format(index_name)] = test_std_y.inverse_transform(
        model.predict(poly.fit_transform(test_std_x.transform(data[NORMAL_ITEMS])))
    ).reshape(-1, 1)
    print(data.shape)

    return abs(data[['{}_hat'.format(index_name)]])


def test():
    """
    Test.
    """
    session = orm.create_session()
    # data query
    train_start = datetime(2018, 5, 1, 0)
    train_end = datetime(2020, 8, 31, 20)
    predict_start = datetime(2018, 5, 1, 0)
    predict_end = datetime(2020, 8, 30, 20)
    station = "龙门大桥"
    data_train = query_data(session=session, station_name=station, start=train_start, end=train_end)
    data_predict = query_data(session=session, station_name=station, start=predict_start, end=predict_end)
    print("Train Size: {}".format(data_train.shape))
    print("Predict Size: {}".format(data_predict.shape))
    # model param
    indexs = model_params['indexs']
    model = model_params['model']
    modes = model_params['modes']

    if modes == 'train':
        # multiprocessing
        threads = []
        for _, index in enumerate(indexs):
            worker = WorkThread(train, (model,index,data_train), 'train_{}'.format(index))
            threads.append(worker)
        for i in range(len(indexs)):
            threads[i].start()
        for i in range(len(indexs)):
            threads[i].join()

    elif modes == 'predict':
        # multiprocessing
        threads = []
        results = []
        for _, index in enumerate(indexs):
            worker = WorkThread(predict, (model,index,data_predict), 'predict_{}'.format(index))
            threads.append(worker)
        for i in range(len(indexs)):
            threads[i].start()
        for i in range(len(indexs)):
            threads[i].join(3)
            results.append(threads[i].get_result()) 

#         print(data_predict)
        predictions = pd.concat(results, axis=1)
        data_insert = pd.concat([data_predict, predictions], axis=1)
        # insert
        print(data_insert)
        insert_data(session=session, station_name=station, model_name=model, data=data_insert)
        
        session.commit()

    else:
        print('Wrong Modes With [ {} ]'.format(modes))
        sys.exit()
        
    session.close()


if __name__ == "__main__":
    test()