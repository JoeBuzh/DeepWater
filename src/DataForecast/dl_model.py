# -*- encoding: utf-8 -*-
'''
@Filename    : dl_model.py
@Datetime    : 2020/09/14 14:20:50
@Author      : Joe-Bu
@version     : 1.0
'''

""" 深度学习预测建模工具 """

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf


def feature_time(data: pd.DataFrame) -> pd.DataFrame:
    """
    Time Feature Engineering.
    """
#     print(data)
#     print(data.info())
    day = 24*60*60
    year = (365.2425)*day
    time_stp = data['time'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:00") if isinstance(x, str) else x
    ).map(datetime.timestamp)
    
    data['day_sin'] = np.sin(time_stp * (2*np.pi / day))
    data['day_cos'] = np.cos(time_stp * (2*np.pi / day))
    data['year_sin'] = np.sin(time_stp * (2*np.pi / year))
    data['year_cos'] = np.cos(time_stp * (2*np.pi / year))
    
    return data


def train_val_test_split(data: pd.DataFrame, train_size: float, val_size: float) -> tuple:
    """
    Time-Series Data Split. 
    """
    n = len(data)
    train_df = data[: int(n*train_size)]
    val_df = data[int(n*train_size): int(n*(train_size+val_size))]
    test_df = data[int(n*(train_size+val_size)):]
    
    return train_df, val_df, test_df


def model_train(model, window, patience, max_epochs):
    """
    Complie & Fit.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode="min")
    
    history = model.fit(
        window.train, 
        epochs=max_epochs,
        validation_data=window.val,
        callbacks=[early_stopping])
    
    return history, model


def model_predict(model, inputs):
    """
    Predict.
    """
#     print(inputs)
    return model.predict(
        x=inputs, 
        batch_size=None, 
        verbose=0, 
        steps=None,
        workers=5,
        use_multiprocessing=False,)


def model_save(model, savepath:str):
    """
    Save Trained Model.
    """
    if not os.path.exists(savepath):
        model.save(savepath)
        assert os.path.exists(savepath)
        print('Model Save Done.')
        
    else:
        print('Model Already Existed.')
    
    
def model_load(modelpath: str):
    """
    Load Saved Model.
    """
    assert os.path.exists(modelpath)
    
    return tf.keras.models.load_model(modelpath)


