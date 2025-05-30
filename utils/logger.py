import logging
import os

def setup_logger(log_path=None, name="default_logger"):
    logger = logging.getLogger(name)

    # Prevent handler duplication
    if logger.hasHandlers():
        logger.handlers.clear()  # <---- forcefully clear previous handlers

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # Stream (console) output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File output (only if path is provided)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
