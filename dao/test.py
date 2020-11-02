# -*- encoding: utf-8 -*-
'''
@Filename    : run.py
@Datetime    : 2020/09/22 16:59:29
@Author      : Joe-Bu
@version     : 1.0
'''

""" ORM Test """

import sys
sys.path.append('../')
import traceback 

import pandas as pd

from dao.orm.SQLite_dynamic import orm
from dao.orm.SQLite_dynamic import AutoStation, MonSection
from dao.orm.SQLite_dynamic import ObsDataRaw, ObsDataQcLinear


def test():
    try:
        session = orm.create_session()
        # orm test
        auto_stat_records = session.query(AutoStation.name, AutoStation.lon, AutoStation.lat).filter_by(provincename='北京市').all()
        moni_sect_records = session.query(MonSection.rb_name, MonSection.riverlak, MonSection.rb_code).filter_by(provincename='河北省').all()
        raw_records = session.query(ObsDataRaw.time, ObsDataRaw.name, ObsDataRaw.pH, ObsDataRaw.codmn).filter_by(time='2020-08-08 12:00:00').all()
        qc_records = session.query(ObsDataQcLinear.time, ObsDataQcLinear.pH, ObsDataQcLinear.tp).filter_by(time='2020-08-08 12:00:00').all()

        print(pd.DataFrame(auto_stat_records))
        print("*"*66)
        print(pd.DataFrame(moni_sect_records))
        print("*"*66)
        print(pd.DataFrame(raw_records))
        print("*"*66)
        print(pd.DataFrame(qc_records))

    except Exception as e:
        traceback.print_exc()


if __name__ == '__main__':
    test()