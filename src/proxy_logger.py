import logging
from logging.handlers import RotatingFileHandler
import os

class LoggerSetup:
    def __init__(self, export_dir: str, log_name: str = "logger", max_bytes: int = 500_000_000, backup_count: int = 1000):
        self.export_dir = export_dir
        self.log_name = log_name
        self.max_bytes = max_bytes
        self.backup_count = backup_count

    def get_logger(self, logger_name: str = "logger") -> logging.Logger:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        log_path = os.path.join(self.export_dir, self.log_name)
        handler = RotatingFileHandler(log_path, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
    
    @staticmethod
    def get_null_logger(name: str = "NullLogger") -> logging.Logger:
        null_logger = logging.getLogger(name)
        null_logger.setLevel(logging.CRITICAL + 1)
        if not null_logger.hasHandlers():
            null_logger.addHandler(logging.NullHandler())
        return null_logger


