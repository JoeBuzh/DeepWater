# -*- encoding: utf-8 -*-
'''
@Filename    : DataQC/run.py
@Datetime    : 2020/10/21 17:52:48
@Author      : Joe-Bu
@version     : 1.0
'''

import os
import sys
sys.path.append("../../")
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import between

from utils.ConstantUtils import MonitorItem
from dao.orm.PostgreSQL_dynamic import orm
from dao.orm.PostgreSQL_dynamic import QualityControlInput, QualityControlOutput


def query_qc_input(session, station_name: str, start: datetime, end: datetime) -> pd.DataFrame:
    time_format = "%Y-%m-%d %H:00:00"
    resp = session.query(
        QualityControlInput.time,
        QualityControlInput.as_code,
        QualityControlInput.watertemp,
        QualityControlInput.pH,
        QualityControlInput.conductivity,
        QualityControlInput.turbidity,
        QualityControlInput.DO,
        QualityControlInput.codmn,
        QualityControlInput.nh3n,
        QualityControlInput.tp,
        QualityControlInput.tn) \
            .filter_by(name=station_name) \
            .filter(between(QualityControlInput.time, start.strftime(time_format), end.strftime(time_format))) \
            .all()
    
    data = pd.DataFrame(resp)
    
    return data.replace([-999.0, 999.0], [np.nan, np.nan])


def insert_qc_output(session, data: pd.DataFrame, station_name: str):
    time_format = "%Y-%m-%d %H:00:00"
    lines = []
    for _, row in data.iterrows():
        line = QualityControlOutput(
            time=row.time,
            name=station_name,
            as_code=row.as_code,
            watertemp=row.watertemp,
            pH=row.pH,
            conductivity=row.conductivity,
            turbidity=row.turbidity,
            DO=row.DO,
            codmn=row.codmn,
            nh3n=row.nh3n,
            tp=row.tp,
            tn=row.tn)
        
        lines.append(line)
        
    session.add_all(lines)
    session.flush()
    session.commit()
    
    
def quality_control(data: pd.DataFrame) -> pd.DataFrame:
    for i, col in enumerate(MonitorItem.ITEMS.value):
#         print(col)
        data.loc[data[col]<data[col].quantile(.05), col] = np.nan
        data.loc[data[col]>data[col].quantile(.95), col] = np.nan
    for i, col in enumerate(MonitorItem.INDEX.value):
#         print(col)
        data.loc[data[col]<data[col].quantile(.05), col] = np.nan
        data.loc[data[col]>data[col].quantile(.95), col] = np.nan
 
    return data.interpolate(method='linear').fillna(method='bfill')


def qc_entrance(station: str, index: str, start: datetime, end: datetime):
    session = orm.create_session()
    
    qc_input = query_qc_input(
        session=session,
        station_name=station, 
        start=start, 
        end=end
    )
    
    print(qc_input)
    qc_output = quality_control(qc_input)
    print(qc_output)
    
    insert_qc_output(
        session=session,
        data=qc_output,
        station_name=station
    )
    
    session.close()