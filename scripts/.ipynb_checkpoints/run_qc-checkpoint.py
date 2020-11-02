# -*- encoding: utf-8 -*-
'''
@Filename    : run_qc.py
@Datetime    : 2020/09/27 18:35:44
@Author      : Joe-Bu
@version     : 1.0
'''

import os
import sys
sys.path.append("../")

from utils.ConfigParseUtils import ConfigParser
from src.DataQC.run import qc_entrance



def main():
    """
    Quality Control Workflow.
    """
    cfp = ConfigParser()
    
    if cfp.start is not None and cfp.end is not None:
        assert cfp.start <= cfp.end
    else:
        print("Error Start or End Time.")
        sys.exit()
        
    if cfp.index != "all":
        print("Index should be [ all ].")
        sys.exit()
        
    if cfp.model == "qc":
        qc_entrance(station=cfp.name, index=cfp.index, start=cfp.start, end=cfp.end)
    else:
        print("Index should be [ qc ].")
        sys.exit()
        

if __name__ == "__main__":
    main()