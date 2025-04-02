#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志模块
"""

import os
import sys
import logging
from pathlib import Path
from loguru import logger

from app.core.config import settings


def setup_logging():
    """配置日志系统"""
    
    # 获取配置中的日志级别
    log_level = settings.LOG_LEVEL.upper()
    
    # 创建日志目录（如果不存在）
    logs_dir = Path(settings.LOG_DIR)
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    # 配置loguru
    config = {
        "handlers": [
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                "level": log_level,
            },
            {
                "sink": logs_dir / "app.log",
                "rotation": settings.LOG_ROTATION,
                "retention": settings.LOG_RETENTION,
                "compression": settings.LOG_COMPRESSION,
                "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                "level": log_level,
                "enqueue": True,
            },
        ],
    }
    
    # 应用配置
    logger.configure(**config)
    
    # 拦截标准库的日志
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # 获取对应的loguru级别
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # 查找调用者
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            # 记录日志
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # 配置标准库日志模块
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # 替换标准库中的Handler
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = [InterceptHandler()]
    
    logger.info(f"日志系统初始化完成，日志级别：{log_level}，日志保存路径：{logs_dir}")
    return logger 
