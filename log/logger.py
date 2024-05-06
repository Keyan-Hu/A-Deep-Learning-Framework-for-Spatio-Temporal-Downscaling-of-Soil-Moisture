import logging

def setup_logger(log_path, append=False):
    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.handlers = []

    logger.setLevel(logging.DEBUG)

    # 根据 append 参数决定是续写模式还是覆盖写入模式
    file_mode = 'a' if append else 'w'
    file_handler = logging.FileHandler(log_path, mode=file_mode)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 创建控制台处理器，只输出警告及以上级别的日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger