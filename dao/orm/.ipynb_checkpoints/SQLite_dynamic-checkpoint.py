# -*- encoding: utf-8 -*-
'''
@Filename    : SQLite.py
@Datetime    : 2020/09/22 11:14:12
@Author      : Joe-Bu
@version     : 1.0
'''

""" SQLite ORM Class """

import sys
sys.path.append("../../")

from sqlalchemy.schema import Table
from sqlalchemy.ext.declarative import declarative_base

from dao.orm.db_orm import DataBaseORM
from config.common import common_params


Base = declarative_base()
orm = DataBaseORM(
    name='SQLite', 
    engine_cmd='sqlite:////{}'.format(common_params.get('SQLite_dir'))
)


class ObsDataRaw(Base):
    __tablename__ = "obs_data_raw_4h"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(ObsDataRaw.__tablename__)


class AutoStation(Base):
    __tablename__ = "automatic_station"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(AutoStation.__tablename__)


class MonSection(Base):
    __tablename__ = "monitor_section"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(MonSection.__tablename__)


class ObsDataQcLinear(Base):
    __tablename__ = "obs_data_qc_linear_4h"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(ObsDataQcLinear.__tablename__)


class ObsDataQcNonLinear(Base):
    __tablename__ = "obs_data_qc_nonlinear_4h"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(ObsDataQcNonLinear.__tablename__)


class ForecastArima(Base):
    __tablename__ = "forecast_arima"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastArima.__tablename__)


class ForecastFbprophet(Base):
    __tablename__ = "forecast_fbprophet"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastFbprophet.__tablename__)


class ForecastLSTM(Base):
    __tablename__ = "forecast_lstm"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastLSTM.__tablename__)


class ForecastGRU(Base):
    __tablename__ = "forecast_gru"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastGRU.__tablename__)


class ForecastEnsemble(Base):
    __tablename__ = "forecast_lstm"
    __table__ = Table(__tablename__, orm.matedata, autoload=True, autoload_with=orm.engine)
    
    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastEnsemble.__tablename__)