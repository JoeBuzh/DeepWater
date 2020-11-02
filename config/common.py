# -*- encoding: utf-8 -*-
'''
@Filename    : common.py
@Datetime    : 2020/09/28 10:48:18
@Author      : Joe-Bu
@version     : 1.0
'''

""" Common Configurations """

import os
import sys


if sys.platform == "darwin":
    proj_home = r'/Users/joe/Desktop/WaterProject/DeepWater'
elif sys.platform == "linux":
    proj_home = r'/public/home/buzh/water/DeepWater'


common_params = dict(
    FREQ = 4,
    # SQLite
    SQLite_dir = os.path.join(proj_home, "data/waterTestDB.db"),
    # PostgreSQL
    PostgreSQL_info = dict(
        username = 'vistor_01',
        password = 'vistor_01',
        hostname = '10.110.18.200',
        port = '5061',
        dbname = 'statistical_prediction_sample_db',
    ),   # TODO: Change
)


forecast_params = dict(
    # Forecast Common
    FREQ = 4,
    LENS = 42,
    INDICES=['nh3n'],
    # INDICES=['codmn', 'nh3n', 'tp', 'tn']
    MODELS=['Arima', 'Fbprophet', 'LSTM', 'GRU'],
    STATIONS=['连江荷山渡口'],
    #STATIONS=['南溪浮宫桥','连江荷山渡口','长汀美溪桥']
    LOGS_HOME=os.path.join(proj_home, 'logs/logs_forecast'),
    
    # Airma
    arima_params = dict(
        last_days = 30,
    ),
    
    # Fbprophet
    fbprophet_params = dict(
        last_days = 30,
    ),
    
    # LSTM
    lstm_params = dict(
        last_days = 365,
        MAX_EPOCH = 10, 
        BATCH = 32,
        IN_STEPS = 42,
        OUT_STEPS = 42,
    ),

    # GRU
    gru_params = dict(
        last_days = 365,
        MAX_EPOCH = 10, 
        BATCH = 32,
        IN_STEPS = 60,
        OUT_STEPS = 42,
    ),
)


# TODO
simulate_params = dict(
    savedir = os.path.join(proj_home, 'model'),
    modes = 'predict',                      # train/predict
    index = 'codmn',                        # TP/TN/NH3N/CODMn
    indexs = ['codmn','nh3n','tp','tn'],    # TP+TN+NH3N+CODMn
    model = 'XGB',                          # RF/GBRT/XGB
    models = ['RF','GBRT','XGB'],
)


# TODO
diagnose_params = dict(
    a=3,
    b=3,
    c=3,
)
