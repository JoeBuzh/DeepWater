# -*- encoding: utf-8 -*-
'''
@Filename    : PostgreSQL_dynamic.py
@Datetime    : 2020/10/21 11:33:50
@Author      : Joe-Bu
@version     : 1.0
'''

""" PostgreSQL ORM Class """

import sys
sys.path.append("../../")

from sqlalchemy.schema import Table
from sqlalchemy.ext.declarative import declarative_base

from dao.orm.db_orm import DataBaseORM
from config.common import common_params


Base = declarative_base()
orm = DataBaseORM(
    name='PostgreSQL',
    engine_cmd='postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}'.format(
        common_params['PostgreSQL_info']['username'],
        common_params['PostgreSQL_info']['password'],
        common_params['PostgreSQL_info']['hostname'],
        common_params['PostgreSQL_info']['port'],
        common_params['PostgreSQL_info']['dbname'],
    )
)


class QualityControlInput(Base):
    __tablename__ = "quality_control_input"
    __table__ = Table(
        __tablename__,
        orm.matedata,
        autoload=True,
        autoload_with=orm.engine
    )

    def __repr__(self) -> str:
        return "table:\t{}".format(QualityControlInput.__tablename__)


class QualityControlOutput(Base):
    __tablename__ = "quality_control_output"
    __table__ = Table(
        __tablename__,
        orm.matedata,
        autoload=True,
        autoload_with=orm.engine
    )

    def __repr__(self) -> str:
        return "table:\t{}".format(QualityControlOutput.__tablename__)


class ForecastArima(Base):
    __tablename__ = "forecast_arima"
    __table__ = Table(
        __tablename__,
        orm.matedata,
        autoload=True,
        autoload_with=orm.engine
    )

    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastArima.__tablename__)


class ForecastFbprophet(Base):
    __tablename__ = "forecast_fbprophet"
    __table__ = Table(
        __tablename__,
        orm.matedata,
        autoload=True,
        autoload_with=orm.engine
    )

    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastFbprophet.__tablename__)


class ForecastLSTM(Base):
    __tablename__ = "forecast_lstm"
    __table__ = Table(
        __tablename__,
        orm.matedata,
        autoload=True,
        autoload_with=orm.engine
    )

    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastLSTM.__tablename__)


class ForecastGRU(Base):
    __tablename__ = "forecast_gru"
    __table__ = Table(
        __tablename__,
        orm.matedata,
        autoload=True,
        autoload_with=orm.engine
    )

    def __repr__(self) -> str:
        return "table:\t{}".format(ForecastGRU.__tablename__)