# -*- encoding: utf-8 -*-
'''
@Filename    : stramlit_demo.py
@Datetime    : 2020/10/30 14:05:28
@Author      : Joe-Bu
@version     : 1.0
'''

import os 
import sys
sys.path.append("../../")

import numpy as np
import pandas as pd
import streamlit as st

from dao.orm.SQLite_dynamic import ObsDataRaw, ObsDataQcLinear, AutoStation
from dao.orm.SQLite_dynamic import ForecastArima, ForecastFbprophet, ForecastLSTM, ForecastGRU


