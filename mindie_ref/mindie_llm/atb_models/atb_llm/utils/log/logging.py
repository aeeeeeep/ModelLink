#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2018. All rights reserved
"""
logging
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from .multiprocess_logging_handler import install_logging_handler

# LOG_LEVEL
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# 日志保存到文件,默认为空，
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "")


def init_logger(logger_ins: logging.Logger, file_name: str):
    """
    日志初始化
    :param logger:
    :param file_name:
    :return:
    """
    logger_ins.setLevel(logging.getLevelName(LOG_LEVEL))
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [pid: %(process)d] %(filename)s-%(lineno)d: %(message)s'
    )
    if file_name:
        # 创建日志记录器，指明日志保存路径,每个日志的大小，保存日志的上限
        file_handle = RotatingFileHandler(
            filename=file_name,
            maxBytes=int(os.getenv('PYTHON_LOG_MAXSIZE', "1073741824")),
            backupCount=10)

        # 将日志记录器指定日志的格式
        file_handle.setFormatter(formatter)
        # 为全局的日志工具对象添加日志记录器
        logger_ins.addHandler(file_handle)

    # 添加控制台输出日志
    console_handle = logging.StreamHandler()
    console_handle.setFormatter(formatter)
    logger_ins.addHandler(console_handle)
    install_logging_handler(logger_ins)
    return logger_ins


logger = init_logger(logging.getLogger(), LOG_TO_FILE)
