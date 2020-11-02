# -*- encoding: utf-8 -*-
'''
@Filename    : evaluate.py
@Datetime    : 2020/09/02 09:05:54
@Author      : Joe-Bu
@version     : 1.0
'''

import os
import sys
sys.path.append('../../')
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.DataForecast.settings import paths, filename, start, end
from src.DataForecast.settings import INDEX, SCALE, FREQ, EVALUATE_SCALE
from src.DataForecast.utils import read_data, read_forecast
from src.DataForecast.utils import checkdir, save_data


class BackEvaluate:
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
        

def calc_accuracy():
    pass
        
        
def calc_mae(y_true, y_pred) -> float:
    """
    Calculate Mean Absolute Error.
        y_true: Ture Observation.
        y_pred: Forecast Value.
    """
    return mean_absolute_error(y_true, y_pred)
    
    
def calc_mse(y_true, y_pred) -> float:
    """
    Calculate Mean Squared Error.
        y_true: Ture Observation.
        y_pred: Forecast Value.
    """
    return mean_squared_error(y_true, y_pred)


def calc_nash(y_true, y_pred) -> float:
    """
    Calculate Nash Correlation Index.
        y_true: Ture Observation.
        y_pred: Forecast Value.
    """
    return r2_score(y_true, y_pred)
    
    
def load_data(paths: dict, filename: str) -> tuple:
    """
    Load Truth & Forecast Data.
    """
    truth_df = read_data(
        os.path.join(paths['in_path'], filename)
    )
    truth_index = truth_df[['Time', INDEX]].set_index('Time')

    pred_df = read_forecast(
        os.path.join(paths['in_path'], '{}_predictions.xls'.format(INDEX))
    )
    pred_df.index.name = 'Time'
    
    return truth_index, pred_df


def main():
    t_df, p_df = load_data(paths, filename)

    back_start = start + timedelta(hours=FREQ*SCALE)
    # evaluate_end = evaluate_start
    back_end = end + timedelta(hours=FREQ*SCALE)

    evaluator = BackEvaluate(t_df, p_df, back_start, back_end, paths['out_eval'])
#     evaluator = ForwardEvaluate(t_df, p_df, start, end, paths['out_eval'])
    evaluator.evaluate_duration()


if __name__ == "__main__":
    main()
