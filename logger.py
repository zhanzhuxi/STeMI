import datetime
import logging


def get_logger(path):
    """input logger save path"""
    logger_name = "Main-logger"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path, encoding='utf-8')
    fmt = "[%(asctime)s %(levelname)s %(filename)s line%(lineno)d %(process)d] %(message)s"
    console_handler.setFormatter(logging.Formatter(fmt))
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


now = datetime.datetime.now()
filename = now.strftime('log_%Y-%m-%d-%H-%M-%S.log')
logger = get_logger(filename)
