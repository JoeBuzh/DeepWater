# -*- encoding: utf-8 -*-
'''
@Filename    : ConfigParseUtils.py
@Datetime    : 2020/09/27 17:38:25
@Author      : Joe-Bu
@version     : 1.0
'''

import argparse
from argparse import OPTIONAL
from datetime import datetime


class ConfigParser:
    """
    Command Line Parameters Parser Utility.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "-n", type=str, default=None, help="forecast station name")
        self.parser.add_argument(
            "-m", type=str, default=None, help="forecast method [Arima|Fbprophet|LSTM|Ensemble] ")
        self.parser.add_argument(
            "-i", type=str, default=None, help="forecast index [codmn|nh3n|tp|tn] ")
        self.parser.add_argument(
            "-s", type=str, default=None, help="forecast start time [%Y%m%d%H]")
        self.parser.add_argument(
            "-e", type=str, default=None, help="forecast end time [%Y%m%d%H]")
        self.args = self.parser.parse_args()

    @property
    def name(self) -> str:
        return self.args.n

    @property
    def model(self) -> str:
        return self.args.m
    
    @property
    def index(self):
        return self.args.i

    @property
    def start(self):
        return datetime.strptime(self.args.s, "%Y%m%d%H") if self.args.s is not None else None

    @property
    def end(self):
        return datetime.strptime(self.args.e, "%Y%m%d%H") if self.args.e is not None else None
