# -*- encoding: utf-8 -*-
'''
@Filename    : DataWindow.py
@Datetime    : 2020/09/14 14:20:50
@Author      : Joe-Bu
@version     : 1.0
'''

""" 数据滑动窗口 """

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


class WindowGenerator():
    """
    Data Window Generator.
    """
    def __init__(self, input_width, label_width, shift, batch, train_df, val_df, test_df, label_cols=None):
        # data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # label
        self.label_cols = label_cols
        if label_cols is not None:
            self.label_col_indices = {col:i for i, col in enumerate(label_cols)}
        self.col_indices = {col: i for i, col in enumerate(train_df.columns)}
        # Window parmas
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch = batch
        self.total_width = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_width)[self.input_slice]
        
        self.label_start = self.total_width - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_width)[self.label_slice]
        
    def __repr__(self):
        '''
        Self Information.
        '''
        return '\n\n'.join([
            f'Total Width:   {self.total_width}',
            f'Input Indices: {self.input_indices}',
            f'Label Indices: {self.label_indices}',
            f'Label Column Names: {self.label_cols}'
        ])
    
    def split_window(self, features):
        '''
        Features & Labels Split.
        '''
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]
        if self.label_cols is not None:
            labels = tf.stack([labels[:, :, self.col_indices[col]] for col in self.label_cols], axis=-1)
            
        # -> tf.data.Dataset
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def make_dataset(self, data):
        '''
        Transform to tf.data.Dataset.
        '''
        data = np.array(data, dtype=np.float32)
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_width,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch,)
        dataset = dataset.map(self.split_window)
        
        return dataset
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def example(self):
        '''
        Get Batch Inputs&Labels Example.
        '''
        result = getattr(self, '_example', None)
        if result is not None:
            # # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
            
        return result