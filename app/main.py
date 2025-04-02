#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文生图Agent主应用
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 先设置日志
from app.utils.logging.logger import setup_logging
logger = setup_logging()

# 导入配置和路由
from app.core.config import settings
from app.api.endpoints import router as api_router

# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="基于LLM的文生图智能体",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """健康检查端点"""
    return {
        "status": "ok", 
        "message": f"{settings.PROJECT_NAME}服务正常运行",
        "environment": "development" if settings.DEBUG else "production"
    }

# 日志系统启动信息
logger.info(f"{settings.PROJECT_NAME} 应用已启动")
logger.debug(f"调试模式: {settings.DEBUG}")
logger.debug(f"API前缀: {settings.API_V1_STR}")
logger.debug(f"CORS设置: {settings.BACKEND_CORS_ORIGINS}") 