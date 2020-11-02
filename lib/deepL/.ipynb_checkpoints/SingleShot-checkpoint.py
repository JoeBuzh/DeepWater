# -*- encoding: utf-8 -*-
'''
@Filename    : SingleShot.py
@Datetime    : 2020/09/14 14:20:50
@Author      : Joe-Bu
@version     : 1.0
'''

""" 单步预测7天48时次模型 """

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers


def multi_step_dense(neural_nums: int, out_steps: int, dropout_rate: float, num_features: int):
    """
    Multi Dense Layers Model.
    """
    model = tf.keras.Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(neural_nums, activation='relu'),
        # dropout
        tf.keras.layers.Dropout(rate=dropout_rate),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(out_steps*num_features, activation='relu'),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, num_features])
    ],
    name='DNN')
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    
    return model    


def multi_step_conv(neural_nums: int, out_steps: int, dropout_rate: float, num_features: int):
    """
    Conv.
    """
    CONV_WIDTH = 12
    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(neural_nums, activation='relu', kernel_size=(CONV_WIDTH)),
        # dropout
        tf.keras.layers.Dropout(rate=dropout_rate),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(out_steps*num_features, activation='relu'),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, num_features])
    ],
    name='CNN')
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    
    return model


def multi_step_lstm(neural_nums: int, out_steps: int, dropout_rate: float, num_features: int):
    """
    LSTM.
    """
    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(neural_nums, return_sequences=False),
        # dropout
        tf.keras.layers.Dropout(rate=dropout_rate),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(out_steps*num_features, activation='relu'),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, num_features])],
        name='LSTM'
    )
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    
    return model