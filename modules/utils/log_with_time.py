import logging


def log_with_time(message: str, level: str = 'INFO'):
    """
    记录带有时间戳的日志消息。

    Args:
        message (str): 日志消息内容。
        level (str): 日志级别。默认为 'INFO'，支持 'INFO', 'ERROR', 'DEBUG' 等。
    """
    # 配置日志格式，包含时间戳和日志级别
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=log_format, level=logging.DEBUG)

    # 根据日志级别记录日志
    if level == 'INFO':
        logging.info(message)
    elif level == 'ERROR':
        logging.error(message)
    elif level == 'DEBUG':
        logging.debug(message)
    else:
        logging.warning(f"未知的日志级别: {level}. 默认使用 INFO 级别.")
        logging.info(message)
