# -*- encoding: utf-8 -*-
'''
@Filename    : back_evaluate.py
@Datetime    : 2020/09/15 20:18:07
@Author      : Joe-Bu
@version     : 1.0
'''

""" 逆序实时评价功能类 """

import os
import sys
sys.path.append('../../')
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.DataEvaluate.settings import *
from src.DataEvaluate.utils import *


class BackEvaluate(object):
    """
    Forecast Data Back Evaluation Class.
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

    def back_search(self, this_time: datetime) -> dict:
        """
        Evaluate Forecast-Index in One Moment.
        """
        group = dict()
        group[this_time] = self.real_df[:this_time.strftime("%Y-%m-%d %H:00:00")][INDEX].values[-1]
        print(group)

        for i in range(SCALE):
            this_time -= timedelta(hours=FREQ)
            group[this_time] = self.pred_df[:this_time.strftime("%Y-%m-%d %H:00:00")][i].values[-1]

        return group 

    def evaluate_duration(self, interval:int=1):
        """
        Evaluate From Start to End.
        Interval: Evaluate Step.
        """
        eval_start = self.start
        eval_end = self.end
        print(eval_start, eval_end)

        t_list = []
        e_95_list = []
        e_max_list = []

        while eval_start <= eval_end:
            print('Evaluating: {}'.format(eval_start.strftime("%Y-%m-%d %H")))
            data_dict = self.back_search(eval_start)
            this_data = [[] for i in range(2)]
            this_data[0] = [x for x in data_dict.keys()]
            this_data[1] = [x for x in data_dict.values()]
            temp = pd.DataFrame(
                {'Time': this_data[0], INDEX: this_data[1]}
            ).replace(-999, np.nan).iloc[::-1].set_index('Time')
            this_val = temp[INDEX][-1]
            temp['err_rate'] = abs(temp[INDEX]-this_val) / this_val
            err_95 = temp['err_rate'][-EVALUATE_SCALE:].quantile(.95)
            err_max = temp['err_rate'][-EVALUATE_SCALE:].max()
            print('err 95% {0}, err max: {1}'.format(err_95, err_max))
            # 
            t_list.append(eval_start)
            e_95_list.append(err_95)
            e_max_list.append(err_max)
            # print(temp)
            save_data(
                temp, 
                os.path.join(self.save_dir, '{0}_{1}_Back_Evaluation.xls'.format(temp.index[-1], INDEX))
            )
            self.back_plot(temp)
            # Time Strding by interval*FREQ
            eval_start += timedelta(hours=interval*FREQ)
        
        err_df = pd.DataFrame(
            {'Time': t_list, 'Err_95': e_95_list, 'Err_max': e_max_list}
        ).set_index('Time')
        save_data(
            err_df, 
            os.path.join(self.save_dir, '{0}_{1}_{2}_{3}_Forecast_Error.xls'.format(
                err_df.index[0], err_df.index[-1], INDEX, EVALUATE_SCALE))
        )
        self.error_plot(err_df)

    def error_plot(self, data: pd.DataFrame):
        """
        Plot Error Rate.
        """
        plt.figure(figsize=(20,5))
        plt.plot(data.index, data['Err_95'], label='error_95%', color='g')
        plt.plot(data.index, data['Err_max'], label='error_max', color='b')
        plt.axhline(y=0.3, label='Threshold', color='r', linestyle='--')
        plt.legend(loc='best')
        plt.title("{0} to {1} ({2}) Forecast Error".format(data.index[0], data.index[-1], INDEX), fontsize=15)
        plt.savefig(os.path.join(
            self.save_dir,
            '{0}_{1}_{2}_{3}_Forecast_Error.png'.format(data.index[0], data.index[-1], INDEX, EVALUATE_SCALE)
        ))

    def back_plot(self, data: pd.DataFrame):
        """
        Plot One Moment Back Search Forecast.
        """
        plt.figure(figsize=(20,5))
        plt.scatter(data.index, data[INDEX], label='Forecast', color='b')
        plt.axhline(y=data[INDEX][-1], label='Truth', color='g', linestyle='--')
        plt.legend(loc='best')
        plt.title("{0} ({1}) Back Evaluation".format(data.index[-1], INDEX), fontsize=15)
        plt.savefig(os.path.join(
            self.save_dir,
            '{0}_{1}_Back_Evaluation.png'.format(data.index[-1], INDEX)
        ))
