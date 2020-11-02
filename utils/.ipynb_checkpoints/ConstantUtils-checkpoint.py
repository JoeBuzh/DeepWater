# -*- encoding: utf-8 -*-
'''
@Filename    : ConstantUtils.py
@Datetime    : 2020/09/27 17:10:12
@Author      : Joe-Bu
@version     : 1.0
'''

from enum import Enum


class MonitorItem(Enum):
    """
    Water Monitoring Items.
    """
    ITEMS = ['watertemp', 'pH', 'DO', 'conductivity', 'turbidity']
    INDEX = ['codmn', 'nh3n', 'tp', 'tn']

    @classmethod
    def items_len(cls) -> int:
        return len(MonitorItem.ITEMS.value)
    
    @classmethod
    def index_len(cls) -> int:
        return len(MonitorItem.INDEX.value)