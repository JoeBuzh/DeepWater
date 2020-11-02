# -*- encoding: utf-8 -*-
'''
@Filename    : run.py
@Datetime    : 2020/08/31 16:20:50
@Author      : Joe-Bu
@version     : 1.0
'''

""" 预测 """

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('../../')
import warnings
import traceback
from datetime import datetime, timedelta

import numpy as np  
import pandas as pd
from sqlalchemy import between
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config.common import forecast_params

# from dao.orm.SQLite_dynamic import orm, ObsDataQcLinear
# from dao.orm.SQLite_dynamic import ForecastArima, ForecastFbprophet, ForecastLSTM, ForecastGRU
from dao.orm.PostgreSQL_dynamic import orm, QualityControlOutput
from dao.orm.PostgreSQL_dynamic import ForecastArima, ForecastFbprophet, ForecastLSTM, ForecastGRU

from lib.statistic.arima.ARIMA import ArimaModel
from lib.statistic.fbprophet.fbprophet import ProphetModel
from lib.deepL.DataWindow import WindowGenerator
from lib.deepL.SingleShot import multi_step_dense, multi_step_conv, multi_step_lstm, multi_step_gru

from src.DataForecast.st_model import adf_calc, adf_diff, get_order, resid_check
from src.DataForecast.dl_model import feature_time, train_val_test_split
from src.DataForecast.dl_model import model_train, model_predict, model_save, model_load

from utils.LogUtils import init_logger
from utils.ConstantUtils import MonitorItem



def arima_model(data) -> np.array:
    """
    ARIMA Modeling & Forecasting.
    """
    # ADF
    d1 = adf_calc(data)
    d2 = adf_diff(data)
    d = max(d1, d2, 1)
    # BIC
    p, q = get_order(data)
    # ARIMA
    arima = ArimaModel(data, p, d, q)
    arima.show_params()
    print(arima.data.tail(3))
    arima._define()
    model = arima._train()

    return model.forecast(forecast_params['LENS'])[0]


def fbprophet_model(data, this_time, index) -> np.array:
    """
    FbProphet Modeling & Forecasting.
    """
    controller = ProphetModel(
        periods=forecast_params['LENS'], 
        freq=forecast_params['FREQ'], 
        index=index
    )
    model_inited = ProphetModel._define(
        growth='linear', 
        last_days=forecast_params['fbprophet_params']['last_days'], 
        n_chg_points=100
    )
    model_fitted, y_mean, y_std = ProphetModel._train(model_inited, data, index)
    time_seq = controller._make_timestamps(model_fitted)
    forecast = ProphetModel._predict(model_fitted, time_seq, y_mean, y_std)
    
    return forecast.loc[forecast.ds>this_time.strftime("%Y-%m-%d %H:00:00")]['yhat'].values


def deep_train(data: pd.DataFrame, mode: str, kernel: str):
    data = feature_time(data)
    data = data.set_index('time')
    train_df, val_df, test_df = train_val_test_split(data=data, train_size=0.7, val_size=0.2)
    print(train_df.shape)
    print(data.sample(5))
    train_df = (train_df - train_df.mean()) / train_df.std()
    val_df = (val_df - train_df.mean()) / train_df.std()
    test_df = (test_df - train_df.mean()) / train_df.std()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(MonitorItem.INDEX.value)
    feature_nums = len(MonitorItem.INDEX.value)
    
    win = WindowGenerator(
        input_width=forecast_params['{0}_params'.format(kernel)]['IN_STEPS'], 
        label_width=forecast_params['{0}_params'.format(kernel)]['OUT_STEPS'], 
        shift=forecast_params['{0}_params'.format(kernel)]['OUT_STEPS'], 
        batch=forecast_params['{0}_params'.format(kernel)]['BATCH'], 
        train_df=train_df, 
        val_df=val_df, 
        test_df=test_df,
        label_cols=MonitorItem.INDEX.value)
    print(win)
    
    if mode == 'dense':
        model = multi_step_dense(
            neural_nums=512, 
            out_steps=forecast_params['{0}_params'.format(kernel)]['OUT_STEPS'], 
            dropout_rate=0.1, num_features=feature_nums)
    elif mode == 'conv':
        model = multi_step_conv(
            neural_nums=256, 
            out_steps=forecast_params['{0}_params'.format(kernel)]['OUT_STEPS'], 
            dropout_rate=0.1, 
            num_features=feature_nums)
    elif mode == 'lstm':
        model = multi_step_lstm(
            neural_nums=32, 
            out_steps=forecast_params['{0}_params'.format(kernel)]['OUT_STEPS'], 
            dropout_rate=0.1, 
            num_features=feature_nums)
    elif mode == 'gru':
        model = multi_step_gru(
            neural_nums=32, 
            out_steps=forecast_params['{0}_params'.format(kernel)]['OUT_STEPS'], 
            dropout_rate=0.1, 
            num_features=feature_nums)
    
    else:
        model = None

    if model is not None:
        _, fitted_model = model_train(
            model=model, 
            window=win, 
            patience=3, 
            max_epochs=forecast_params['{0}_params'.format(kernel)]['MAX_EPOCH'])       
        return fitted_model
    else:
        print('Wrong DL Mode with [ {} ]'.format(mode))
        sys.exit()

    
def deep_predict(data: pd.DataFrame, model, kernel: str):
    """
    Deepl Predict Method.
    """
    data = feature_time(data)
    print(data.shape)
    print(data.tail(5))
    # normalization
    data = data.set_index('time')
    data_norm = (data - data.mean()) / data.std()
    print("name: ", model.name)

    this_predict = model_predict(
        model, 
        data_norm.values.reshape((1, forecast_params['{0}_params'.format(kernel)]['IN_STEPS'], -1))
    )
    print(this_predict[0].shape)

    temp = {}
    for i, col in enumerate(MonitorItem.INDEX.value):
        temp[col] = this_predict[0][:,i] * data[col].std() + data[col].mean()
    this_forecast = pd.DataFrame(temp)

    return this_forecast
    

def deepl_model(this_train: pd.DataFrame, this_predict: pd.DataFrame, kernel: str):
    """
    LSTM Modeling & Forecasting.
    """
    this_forecast = None
    if kernel == "lstm":
        # train
        model = deep_train(data=this_train, mode="lstm", kernel=kernel)
        # predict
        this_forecast = deep_predict(this_predict, model, kernel=kernel)
    elif kernel == "gru":
        # train
        model = deep_train(data=this_train, mode="gru", kernel=kernel)
        # predict
        this_forecast = deep_predict(this_predict, model, kernel=kernel)    

    return this_forecast
    

def insert_arima(session, data: np.array, this_time: datetime, station_name: str, index: str):
    """
    写入ARIMA预报数据
    """
    insert_line = ForecastArima(
        forecast_time=this_time.strftime("%Y-%m-%d %H:00:00"),
        name=station_name,
        param=index
    )
    for i, num in enumerate(data):
        setattr(insert_line, 'forecast_{}'.format(i), num)
    
    if session.query(ForecastArima).filter(ForecastArima.name==station_name).filter(ForecastArima.param==index).filter(ForecastArima.forecast_time==this_time.strftime("%Y-%m-%d %H:00:00")).all():
        session.query(ForecastArima) \
            .filter(ForecastArima.name==station_name) \
            .filter(ForecastArima.param==index) \
            .filter(ForecastArima.forecast_time==this_time.strftime("%Y-%m-%d %H:00:00")) \
            .update({'forecast_{0}'.format(i): val for i, val in enumerate(data)})
        session.commit()
        print('UPDATE {0} Arima data done.'.format(this_time.strftime("%Y-%m-%d %H:00:00")))
    else:
        session.add(insert_line)
        session.flush()
        session.commit()
        print('INSERT {0} Arima data done.'.format(this_time.strftime("%Y-%m-%d %H:00:00")))


def insert_fbprophet(session, data: np.array, this_time: datetime, station_name: str, index: str):
    """
    写入FbPropHet预报数据
    """
    insert_line = ForecastFbprophet(
        forecast_time=this_time.strftime("%Y-%m-%d %H:00:00"),
        name=station_name,
        param=index
    )
    for i, num in enumerate(data):
        setattr(insert_line, 'forecast_{}'.format(i), num)
    
    if session.query(ForecastFbprophet).filter(ForecastFbprophet.name==station_name).filter(ForecastFbprophet.param==index).filter(ForecastFbprophet.forecast_time==this_time.strftime("%Y-%m-%d %H:00:00")).all():
        session.query(ForecastFbprophet) \
            .filter(ForecastFbprophet.name==station_name) \
            .filter(ForecastFbprophet.param==index) \
            .filter(ForecastFbprophet.forecast_time==this_time.strftime("%Y-%m-%d %H:00:00")) \
            .update({'forecast_{0}'.format(i): val for i, val in enumerate(data)})
        session.commit()
        print('UPDATE {0} Fbprophet data done.'.format(this_time.strftime("%Y-%m-%d %H:00:00")))
    else:
        session.add(insert_line)
        session.flush()
        session.commit()
        print('INSERT {0} Fbprophet data done.'.format(this_time.strftime("%Y-%m-%d %H:00:00")))


def insert_lstm(session, data: pd.DataFrame, this_time: datetime, station_name: str):
    """
    写入LSTM预报数据
    """
    lines = []
    print(data)
    for col in data.columns:
        line = ForecastLSTM(
            forecast_time=this_time.strftime("%Y-%m-%d %H:00:00"),
            name=station_name,
            param=col
        )
        for i, num in enumerate(data[col]):
            setattr(line, 'forecast_{}'.format(i), num)
            
        if session.query(ForecastLSTM).filter(ForecastLSTM.name==station_name).filter(ForecastLSTM.param==col).filter(ForecastLSTM.forecast_time==this_time.strftime("%Y-%m-%d %H:00:00")).all():
            session.query(ForecastLSTM) \
                .filter(ForecastLSTM.name==station_name) \
                .filter(ForecastLSTM.param==col) \
                .filter(ForecastLSTM.forecast_time==this_time.strftime("%Y-%m-%d %H:00:00")) \
                .update({'forecast_{0}'.format(i): val for i, val in enumerate(data[col])})
            session.commit()
            print('UPDATE {0} LSTM data done.'.format(this_time.strftime("%Y-%m-%d %H:00:00")))
        else:
            lines.append(line)
    
    if len(lines) > 0:
        session.add_all(lines)
        session.flush()
        session.commit()
        print('INSERT {0} LSTM data done.'.format(this_time.strftime("%Y-%m-%d %H:00:00")))


def insert_gru(session, data: pd.DataFrame, this_time: datetime, station_name: str):
    """
    写入GRU预报数据
    """
    lines = []
    print(data)
    for col in data.columns:
        line = ForecastGRU(
            forecast_time=this_time.strftime("%Y-%m-%d %H:00:00"),
            name=station_name,
            param=col
        )
        for i, num in enumerate(data[col]):
            setattr(line, 'forecast_{}'.format(i), num)
            
        if session.query(ForecastGRU).filter(ForecastGRU.name==station_name).filter(ForecastGRU.param==col).filter(ForecastGRU.forecast_time==this_time.strftime("%Y-%m-%d %H:00:00")).all():
            session.query(ForecastGRU) \
                .filter(ForecastGRU.name==station_name) \
                .filter(ForecastGRU.param==col) \
                .filter(ForecastGRU.forecast_time==this_time.strftime("%Y-%m-%d %H:00:00")) \
                .update({'forecast_{}'.format(i): val for i, val in enumerate(data[col])})
            session.commit()
            print('UPDATE {0} GRU data done.'.format(this_time.strftime("%Y-%m-%d %H:00:00")))
        else:
            lines.append(line)
        
    if len(lines) > 0:
        session.add_all(lines)
        session.flush()
        session.commit()
        print('INSERT {0} GRU data done.'.format(this_time.strftime("%Y-%m-%d %H:00:00")))

    
def query_obs_4h(session, station_name: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    SQLite 读取 & 解析数据.
    """
    time_format = "%Y-%m-%d %H:00:00"
    resp = session.query(
        ObsDataQcLinear.time,
        ObsDataQcLinear.watertemp,
        ObsDataQcLinear.pH,
        ObsDataQcLinear.DO,
        ObsDataQcLinear.conductivity,
        ObsDataQcLinear.turbidity,
        ObsDataQcLinear.codmn,
        ObsDataQcLinear.nh3n,
        ObsDataQcLinear.tp,
        ObsDataQcLinear.tn) \
            .filter_by(name=station_name) \
            .filter(between(ObsDataQcLinear.time, start.strftime(time_format), end.strftime(time_format))) \
            .all()
    
    data = pd.DataFrame(resp)
    
    return data.replace([-999.0, 999.0], [np.nan, np.nan])


def query_qc_4h(session, station_name: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    PostgreSQL 读取 & 解析数据.
    """
    time_format = "%Y-%m-%d %H:00:00"
    resp = session.query(
        QualityControlOutput.time,
        QualityControlOutput.watertemp,
        QualityControlOutput.pH,
        QualityControlOutput.DO,
        QualityControlOutput.conductivity,
        QualityControlOutput.turbidity,
        QualityControlOutput.codmn,
        QualityControlOutput.nh3n,
        QualityControlOutput.tp,
        QualityControlOutput.tn) \
            .filter_by(name=station_name) \
            .filter(between(QualityControlOutput.time, start.strftime(time_format), end.strftime(time_format))) \
            .all()
    
    data = pd.DataFrame(resp)
    
    return data

    
def forecast_arima(station: str, index: str, start: datetime, end: datetime):
    """
    Apply ARIMA Forecast.
    """
    if not os.getenv("LOGS_HOME"):
        log_home = forecast_params['LOGS_HOME']
    else:
        log_home = os.getenv("LOGS_HOME")
        
    arima_logger = init_logger(
        log_path=os.path.join(log_home, 'arima_{0}.log'.format(start.strftime("%Y%m%d%H"))
    ))
    
    session = orm.create_session()

    preds = dict()
    while start <= end:
        this_start = start - timedelta(days=forecast_params['arima_params']['last_days'])
        this_end = start
        #this_resp = query_obs_4h(session=session, station_name=station, start=this_start, end=this_end)
        this_resp = query_qc_4h(session=session, station_name=station, start=this_start, end=this_end)
        this_data = this_resp[['time', index]].set_index('time')

        arima_logger.info("Query [{0}] history data before [{1}] done.".format(station, start.strftime("%Y/%m/%d %H")))
        this_forecast = arima_model(this_data)
        print("*** {0} Forecast ***\n{1}".format(start.strftime("%Y-%m-%d %H:00:00"), this_forecast))
        arima_logger.info("Forecast [{0}] [{1}] [{2}] done.".format(station, start.strftime("%Y/%m/%d %H"), index))
        insert_arima(
            session, data=this_forecast, this_time=start, station_name=station, index=index
        )
        arima_logger.info("ADD NEW [{0}] [{1}] [{2}] forecast data done.".format(station, start.strftime("%Y/%m/%d %H"), index))
        start += timedelta(hours=forecast_params['FREQ'])

    session.close()
    arima_logger.info("[{0}] close session done.".format(station))


def forecast_fbprophet(station: str, index: str, start: datetime, end: datetime):
    """
    Apply FbPropHet Forecast.
    """
    if not os.getenv("LOGS_HOME"):
        log_home = forecast_params['LOGS_HOME']
    else:
        log_home = os.getenv("LOGS_HOME")
        
    fbprophet_logger = init_logger(
        log_path=os.path.join(log_home,'fbprophet_{0}.log'.format(start.strftime("%Y%m%d%H")))
    )
    session = orm.create_session()

    preds = dict()
    while start <= end:
        this_start = start - timedelta(days=forecast_params['fbprophet_params']['last_days'])
        this_end = start
        #this_resp = query_obs_4h(session=session, station_name=station, start=this_start, end=this_end)
        this_resp = query_qc_4h(session=session, station_name=station, start=this_start, end=this_end)
        this_data = this_resp[['time', index]].rename(columns={'time': 'ds', index: 'y'})
        fbprophet_logger.info("Query [{0}] history data before [{1}] done.".format(station, start.strftime("%Y/%m/%d %H")))

        this_forecast = fbprophet_model(this_data, start, index)
        fbprophet_logger.info("Forecast [{0}] [{1}] [{2}]done.".format(station, start.strftime("%Y/%m/%d %H"), index))
        insert_fbprophet(
            session, data=this_forecast, this_time=start, station_name=station, index=index
        )
        fbprophet_logger.info("ADD NEW [{0}] [{1}] [{2}] forecast data done.".format(station, start.strftime("%Y/%m/%d %H"), index))
        start += timedelta(hours=forecast_params['FREQ'])
        
    session.close()
    fbprophet_logger.info("[{0}] close session done.".format(station))


def forecast_lstm(station: str, index: str, start: datetime, end: datetime):
    """
    Apply LSTM Net Forecast.
    """
    if not os.getenv("LOGS_HOME"):
        log_home = forecast_params['LOGS_HOME']
    else:
        log_home = os.getenv("LOGS_HOME")
        
    lstm_logger = init_logger(
        log_path=os.path.join(log_home, 'lstm_{0}.log'.format(start.strftime("%Y%m%d%H")))
    )
    session = orm.create_session()

    preds = dict()
    while start <= end:
        this_start = start - timedelta(days=forecast_params['lstm_params']['last_days'])
        this_end = start
#         this_train = query_obs_4h(
#             session=session, station_name=station, start=this_start, end=this_end
#         )
        this_train = query_qc_4h(
            session=session, station_name=station, start=this_start, end=this_end
        )
    
        this_predict_start = start - timedelta(hours=(forecast_params['lstm_params']['IN_STEPS']-1)*forecast_params['FREQ'])
        this_predict_end = start    
#         this_predict = query_obs_4h(
#             session=session, station_name=station, start=this_predict_start, end=this_predict_end
#         )
        this_predict = query_qc_4h(
            session=session, station_name=station, start=this_predict_start, end=this_predict_end
        )
        print("This Train Shape: {0}".format(this_train.shape))
        print("This Predict Shape: {0}".format(this_predict.shape))
        lstm_logger.info("Query [{0}] data before [{1}] done.".format(station, start.strftime("%Y/%m/%d %H")))
        
        this_forecast = deepl_model(this_train=this_train, this_predict=this_predict, kernel="lstm")
        lstm_logger.info("Forecast [{0}] [{1}] [{2}] done.".format(station, start.strftime("%Y/%m/%d %H"), index))
        
        insert_lstm(
#             session, data=this_forecast, this_time=start, station_name=station
            session, data=this_forecast[[index]], this_time=start, station_name=station
        )
        lstm_logger.info("ADD NEW [{0}] [{1}] [{2}] forecast data done.".format(station, start.strftime("%Y/%m/%d %H"), index))
        start += timedelta(hours=forecast_params['FREQ'])

    session.close()
    lstm_logger.info("[{0}] close session done.".format(station))


def forecast_gru(station: str, index: str, start: datetime, end: datetime):
    """
    Apply LSTM Net Forecast.
    """
    if not os.getenv("LOGS_HOME"):
        log_home = forecast_params['LOGS_HOME']
    else:
        log_home = os.getenv("LOGS_HOME")
        
    gru_logger = init_logger(
        log_path=os.path.join(log_home, 'gru_{0}.log'.format(start.strftime("%Y%m%d%H")))
    )
    session = orm.create_session()

    preds = dict()
    while start <= end:
        this_start = start - timedelta(days=forecast_params['gru_params']['last_days'])
        this_end = start
#         this_train = query_obs_4h(
#             session=session, station_name=station, start=this_start, end=this_end
#         )
        this_train = query_qc_4h(
            session=session, station_name=station, start=this_start, end=this_end
        )
    
        this_predict_start = start - timedelta(hours=(forecast_params['gru_params']['IN_STEPS']-1)*forecast_params['FREQ'])
        this_predict_end = start
#         this_predict = query_obs_4h(
#             session=session, station_name=station, start=this_predict_start, end=this_predict_end
#         )
        this_predict = query_qc_4h(
            session=session, station_name=station, start=this_predict_start, end=this_predict_end
        )
        print("This Train Shape: {0}".format(this_train.shape))
        print("This Predict Shape: {0}".format(this_predict.shape))
        gru_logger.info("Query [{0}] data before [{1}] done.".format(station, start.strftime("%Y/%m/%d %H")))
        
        this_forecast = deepl_model(this_train=this_train, this_predict=this_predict, kernel="gru")
        gru_logger.info("Forecast [{0}] [{1}] [{2}] done.".format(station, start.strftime("%Y/%m/%d %H"), index))
        
        insert_gru(
            session, data=this_forecast[[index]], this_time=start, station_name=station
        )
        gru_logger.info("ADD NEW [{0}] [{1}] [{2}] forecast data done.".format(station, start.strftime("%Y/%m/%d %H"), index))
        start += timedelta(hours=forecast_params['FREQ'])

    session.close()
    gru_logger.info("[{0}] close session done.".format(station))
