# -*- encoding: utf-8 -*-
'''
@Filename    : settings.py
@Datetime    : 2020/08/31 16:18:07
@Author      : Joe-Bu
@version     : 1.0
'''

""" 评价配置文件 """

from datetime import datetime


''' Normal Prams '''
start = datetime(2020, 7, 10, 0, 0)
# start = datetime(2020, 7, 31, 0, 0)
end = datetime(2020, 8, 1, 0, 0)

TZONE = 30
INDEX = 'CODMn'
FREQ = 4
SCALE = 42
EVALUATE_SCALE = 42

# ------------------------

''' Path Params '''

paths = dict(
    in_path = r'../../output',
    out_path = r'../../output/',
    out_eval = r'../../output/evaluate/{0}/{1}/'.format(INDEX, EVALUATE_SCALE),
    model_path = r'../../model'
)
# filename = u'沿江渡4h验证原始数据-线性扩展.xls'
filename = u'洛宁长水4h验证原始数据-线性扩展.xls'
