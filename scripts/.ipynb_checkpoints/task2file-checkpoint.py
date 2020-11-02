# -*- coding: utf-8 -*-
# Author: Joe-BU

"""
Generate series of task at one specific time.
"""

import os
import sys
sys.path.append("../")

from config.common import forecast_params

INDICES=forecast_params['INDICES']
MODELS=forecast_params['MODELS']
STATIONS=forecast_params['STATIONS']


def check_dir(abspath: str):
    '''
    Check dir exist.
    '''
    if not os.path.exists(abspath):
        os.makedirs(abspath)
        
    assert os.path.exists(abspath)


def main():
    time_str = sys.argv[1]
    proj_home = os.getenv("PROJ_HOME")
    pythonenv = os.getenv("PYTHON")
    logs_home = os.getenv("LOGS_HOME")
    task_home = os.path.join(proj_home, 'task')
    
    check_dir(logs_home)
    check_dir(task_home)
    
    task_file = os.path.join(task_home, 'task_{0}.txt'.format(time_str))
    outs_file = os.path.join(logs_home, 'task_{0}.out'.format(time_str))
    
    if os.path.exists(task_file):
        print("task_{0} already exist, removing ...".format(time_str))
        os.remove(task_file)
    
    with open(task_file, 'w') as f:
        for station in STATIONS:
            for model in MODELS:
                for index in INDICES:
                    cmd = '''cd {0}; {1} run_forecast.py -n '{2}' -m '{3}' -i '{4}' -s '{5}' -e '{5}' >& {6}\n'''.format(
#                     cmd = '''cd {0}; {1} run_forecast.py -n '{2}' -m '{3}' -i '{4}' -s '{5}' -e '{5}'\n'''.format(
                        os.path.join(proj_home, 'scripts'),
                        pythonenv,
                        station,
                        model,
                        index,
                        time_str,   # start & end
                        outs_file)
                    f.write(cmd)
    if os.path.exists(task_file):
        print("task_{} make done!".format(time_str))
    else:
        sys.exit()


if __name__ == "__main__":
    main()
