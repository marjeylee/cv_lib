# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     log_util
   Description :
   Author :       'li'
   date：          2018/9/19
-------------------------------------------------
   Change Activity:
                   2018/9/19:
-------------------------------------------------
"""
import datetime
import os

from utility import combine_file_path
from utility.file_path_utility import create_dir


def get_current_format_time():
    """
    get current time ,eg:2018-09-19 14:03:23.82
    :return:
    """
    now_time = datetime.datetime.now()
    return str(now_time)[0:22]


class LogUtil(object):
    """
    log utility
    """

    def __init__(self, mode='console', log_path=None):
        """
        :param mode: console  or file
        :param log_path:
        """
        assert mode in ['console', 'file']
        self.__mode = mode
        if log_path is None:
            dir_path = combine_file_path('logs')
            create_dir(dir_path)
            log_path = os.path.join(dir_path, 'log.log')
            if not os.path.exists(path=log_path):
                file = open(log_path, encoding='utf8', mode='w')
                file.close()
        self.__log_path = log_path

    def info(self, txt=None):
        """
        info
        :param txt:
        :return:
        """
        self.__log(log_type='INFO', text=txt)

    def debug(self, txt=None):
        """
        info
        :param txt:
        :return:
        """
        self.__log(log_type='DEBUG', text=txt)

    def error(self, txt=None):
        """
        info
        :param txt:
        :return:
        """
        self.__log(log_type='ERROR', text=txt)

    def __log(self, log_type='INFO', text=''):
        """
        log content  ,if mode is console .then print string on console.
        if mode is file,append string to log file.
        :param log_type:
        :param text:
        :return:
        """
        text = '[' + get_current_format_time() + ']' + ' [' + log_type + ']' + ' ' + text

        if self.__mode == 'console':
            print('\033[;33m' + str(text) + '\033[0m')
        else:
            with open(self.__log_path, mode='a', encoding='utf8') as file:
                file.write(text + '\n')
