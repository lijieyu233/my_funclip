import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

def setup_logger():
    # 创建日志记录器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # 创建Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建Formatter -> 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 将Handler添加到日志记录器中
    logger.addHandler(console_handler)

    return logger