# -*- encoding: utf-8 -*-
'''
@Filename    : simulate_settings.py
@Datetime    : 2020/09/24 10:21:18
@Author      : Joe-Bu
@version     : 1.0
'''

NORMAL_ITEMS = ['watertemp', 'pH', 'DO', 'conductivity', 'turbidity']
NORMAL_INDEX = ['codmn', 'nh3n', 'tp', 'tn']

model_params = dict(
    savedir = '../../model',
    modes = 'predict',                      # train/predict
    index = 'codmn',                        # TP/TN/NH3N/CODMn
    indexs = ['codmn','nh3n','tp','tn'],    # TP+TN+NH3N+CODMn
    model = 'XGB',                           # RF/GBRT/XGB
    models = ['RF','GBRT','XGB']            # RF+GBRT+XGB
)