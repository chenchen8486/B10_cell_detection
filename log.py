# coding: utf-8
import logging
from logging import handlers
import os
import json

"""参数配置"""
config_path = './config.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')

# 本地调试路径，包括存图和存日志
SAVE_DETECTION_LOG_PATH = config_data['SAVE_DETECTION_LOG_PATH']

# 日志存储路径
LOG_PATH = os.path.join(SAVE_DETECTION_LOG_PATH, 'logs')
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

ALL_LOG_NAME = 'all.log'
ERROR_LOG_NAME = 'error.log'


class Logger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='midnight', backCount=3,
                 fmt='%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)

        # 设置日志格式
        format_str = logging.Formatter(fmt)

        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))

        # 往屏幕上输出
        sh = logging.StreamHandler()

        # 设置屏幕上显示的格式
        sh.setFormatter(format_str)

        # 往文件里写入, 指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S秒   M分   H小时  D天
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')

        # 设置文件里写入的格式
        th.setFormatter(format_str)

        # 把对象加到logger里
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


# 写日志（所有日志）
ALL_LOG_OBJ = Logger(os.path.join(LOG_PATH, ALL_LOG_NAME), level='info', backCount=30)

# 写日志（仅错误日志）
# ERROR_LOG_OBJ = Logger(os.path.join(LOG_PATH, ERROR_LOG_NAME), level='error', backCount=30)