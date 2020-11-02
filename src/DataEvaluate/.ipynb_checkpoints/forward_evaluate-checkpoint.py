# -*- encoding: utf-8 -*-
'''
@Filename    : forward_evaluate.py
@Datetime    : 2020/09/15 20:18:07
@Author      : Joe-Bu
@version     : 1.0
'''

""" 正向模拟评价功能类 """

import os
import sys
sys.path.append('../../')
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.DataEvaluate.settings import *
from src.DataEvaluate.utils import *


class ForwardEvaluate:
    """
    Forecast Data Foreword Evaluation Class.
    """
    def __init__(
        self, 
        real_df: pd.DataFrame, 
        pred_df: pd.DataFrame, 
        eval_start: datetime, 
        eval_end: datetime, 
        save_dir: str 
    ):
        self.real_df = real_df
        self.pred_df = pred_df
        self.start = eval_start
        self.end = eval_end
        self.save_dir = checkdir(save_dir)
        
    def forward_search(self, this_time: datetime) -> pd.DataFrame:
        """
        Forword Search Ture Value.
        """
        this_start = this_time + timedelta(hours=FREQ)
        this_end = this_time + timedelta(hours=EVALUATE_SCALE*FREQ)
        
        return self.real_df[
            this_start.strftime("%Y-%m-%d %H:00:00"):this_end.strftime("%Y-%m-%d %H:00:00")]
    
    def evaluate_duration(self):
        """
        Evaluate From Start to End.
        """
        eval_start = self.start
        eval_end = self.end
        print(eval_start, eval_end)
        
        time_list = []
        mae_list = []
        mse_list = []
        nash_list = []
        accuracy_list = []
        while eval_start <= eval_end:
            print('Evaluating: {}'.format(eval_start.strftime("%Y-%m-%d %H")))
            this_real = self.forward_search(eval_start)
            this_pred = self.pred_df.loc[self.pred_df.index==eval_start.strftime("%Y-%m-%d %H:00:00")]
            this_real['{}_P'.format(INDEX)] = this_pred.values[0][:EVALUATE_SCALE]
            this_real['err_rate'] = abs(this_real[INDEX]-this_real['{}_P'.format(INDEX)]) / this_real[INDEX]
#             print(this_real)
            mae = calc_mae(this_real[INDEX], this_real['{}_P'.format(INDEX)])
            mse = calc_mse(this_real[INDEX], this_real['{}_P'.format(INDEX)])
            nash = calc_nash(this_real[INDEX], this_real['{}_P'.format(INDEX)])
            accuracy = this_real[this_real['err_rate']<=0.3].shape[0] / this_real.shape[0]
            print(mae, mse, nash, accuracy)
            time_list.append(eval_start)
            mae_list.append(mae)
            mse_list.append(mse)
            nash_list.append(nash)
            accuracy_list.append(accuracy)
#             self.forward_plot(this_real)
            # loop
            eval_start += timedelta(hours=FREQ)
        index_df = pd.DataFrame({
            'Time': time_list, 'mae': mae_list, 'mse': mse_list, 'nash': nash_list, 'accuracy': accuracy_list
        }).set_index('Time')
        print(index_df)
        self.index_plot(index_df)
        print(index_df.describe().T)
        
    def forward_plot(self, data: pd.DataFrame):
        """
        Plot One Moment Forward Search Forecast.
        """
        plt.figure(figsize=(20,5))
        plt.plot(data.index, data[INDEX], label='Forecast', color='b', marker='o', linestyle='--')
        plt.plot(data.index, data['{}_P'.format(INDEX)], label='Truth', color='g', marker='*', linestyle='--')
        plt.legend(loc='best')
        plt.title("{0} ({1}) Forward Evaluation".format(data.index[0], INDEX), fontsize=15)
        plt.savefig(os.path.join(
            self.save_dir,
            '{0}_{1}_Forward_Evaluation.png'.format(data.index[0], INDEX)
        ))
        
    def index_plot(self, data: pd.DataFrame):
        """
        Plot Forward Evaluation Indexs.
        """
        fig = plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['nash'], label='Nash', color='b', marker='o')
        plt.ylabel('Nash')
        plt.title("{0} to {1} ({2}) [{3}] Forward Evaluation".format(data.index[0], data.index[0], INDEX, EVALUATE_SCALE), fontsize=15)
        plt.subplot(2, 1, 2)
        plt.plot(data.index, data['accuracy'], label='Accuracy', color='g', marker='*')
        plt.ylabel('Accuracy')
        
#         plt.title("{0} to {1} ({2}) Forward Evaluation".format(data.index[0], data.index[0], INDEX), fontsize=15)
        plt.savefig(os.path.join(
            self.save_dir,
            '{0}_{1}_{2}_{3}_Forward_Evaluation.png'.format(data.index[0], data.index[0], INDEX, EVALUATE_SCALE)))