# -*- encoding: utf-8 -*-
'''
@Filename    : SQLite.py
@Datetime    : 2020/09/22 11:14:12
@Author      : Joe-Bu
@version     : 1.0
'''

""" SQLite ORM Class """

from ast import Str
import os
from os import name
import sys
sys.path.append("../../")
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.schema import Column
from sqlalchemy import Integer, String, Float

from dao.orm.base import Base


class ObsDataRaw(Base):
    """
    Table obs_data_row_4h
    """
    __tablename__ = "obs_data_row_4h"
    # __tableargs__ = {'extend_existing': True}
    # __tableargs__ = {'schema': 'main'}

    id  = Column(Integer, primary_key=True)
    time = Column(String(32))
    name = Column(String(32))
    watertemp = Column(Float)
    pH = Column(Float)
    DO = Column(Float)
    conductivity = Column(Float)
    turbidity = Column(Float)
    codmn = Column(Float)
    nh3n = Column(Float)
    tp = Column(Float)
    tn = Column(Float)

    def __repr__(self):
        return "table:\t{}".format(ObsDataRaw.__tablename__)


class AutoStation(Base):
    """
    Table Automatic Station.
    """
    __tablename__ = "automatic_station"
    # __tableargs__ = {'extend_existing': True}
    # __tableargs__ = {'schema': 'main'}

    id = Column(Integer, primary_key=True)
    code = Column(String(32))
    name = Column(String(32))
    basin = Column(String(32))
    domain = Column(String(32))
    lon = Column(Float)
    lat = Column(Float)
    up_codes = Column()
    conunitcode = Column(String(32))
    provincename = Column(String(32))
    provincecode = Column(Integer)
    cityname = Column(String(32))
    citycode = Column(Integer)
    countyname = Column(String(32))
    countycode = Column(Integer)

    def __repr__(self):
        return "table:\t{}".format(AutoStation.__tablename__)