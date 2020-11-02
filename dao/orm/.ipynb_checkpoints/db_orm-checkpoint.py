# -*- encoding: utf-8 -*-
'''
@Filename    : db_context_holder.py
@Datetime    : 2020/09/21 16:25:04
@Author      : Joe-Bu
@version     : 1.0
'''

import sys
sys.path.append('../../')

from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData
from sqlalchemy.orm import sessionmaker


class DataBaseORM:
    """
    ORM Tool Class.
    """
    __db_charset = 'utf-8'
    matedata = MetaData()

    def __init__(self, name: str, engine_cmd: str):
        self.name = name
        self.engine = create_engine(engine_cmd, encoding=DataBaseORM.__db_charset)
    
    def create_session(self):
        Session =  sessionmaker(bind=self.engine)
        return Session()

    def _test_session(self, test_sql: str):
        assert self.session is not None and self.engine is not None

        with self.engine.connect() as connection:
            result = connection.execute(test_sql)
            for row in result:
                print(row)
