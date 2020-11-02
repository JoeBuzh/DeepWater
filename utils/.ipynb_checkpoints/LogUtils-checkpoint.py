# -*- coding: utf-8 -*-
# Author: Joe-BU
# Date: 2020-10-15 15:00


import os
import logging


def init_logger(log_path: str):
    """
    Init Logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    
    # FileHandler
    handler = logging.FileHandler(log_path)
    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def test():
    logger = init_logger('out.log')
    
    
if __name__ == "__main__":
    test()


