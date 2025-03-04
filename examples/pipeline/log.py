import logging
import colorlog

# 在文件开头添加日志配置
def setup_logger(name):
    """设置日志配置"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 如果logger已经有handler，就不添加新的handler
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 设置颜色配置
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    return logger

