# -*- encoding: utf-8 -*-
'''
@Filename    : ThreadUtils.py
@Datetime    : 2020/09/24 16:57:09
@Author      : Joe-Bu
@version     : 1.0
'''

import threading
from time import ctime

 
class WorkThread(threading.Thread):
    """
    MultiThreading Handle Class.
    """
    def __init__(self, func, args, name=None):
        super().__init__()
        self.name = name
        self.func = func
        self.args = args

    def run(self):
        print('{0} worker started at {1}'.format(self.name, ctime()))
        self.result = self.func(*self.args)
        print('{0} worker stopped at {1}'.format(self.name, ctime()))

    def get_result(self):
        return self.result
