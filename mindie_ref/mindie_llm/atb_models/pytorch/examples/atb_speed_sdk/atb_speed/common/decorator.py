#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2018. All rights reserved.
"""
decorator
"""

import logging
import time
from functools import wraps, partial


class CommonDecorator:
    """
    CommonDecorator
    """

    @staticmethod
    def timing(func=None, *, logger=None, level=logging.INFO):
        """
        函数计时
        :return:
        """
        if logger is None:
            logger = logging.getLogger()
        if func is None:
            # 没有括号的时候args是func，有括号的时候args是None
            return partial(CommonDecorator.timing, logger=logger, level=level)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            wrapper
            :param args:
            :param kwargs:
            :return:
            """
            func_name = func.__name__
            logger.log(level, f"{func_name} start")
            start_time = time.time()
            res = func(*args, **kwargs)
            logger.log(level, f"{func_name} cost {(time.time() - start_time):.5f}s ")
            return res

        return wrapper
