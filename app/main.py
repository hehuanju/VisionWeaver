#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文生图Agent主应用
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

# 先设置日志
from app.utils.logging.logger import setup_logging
logger = setup_logging()

# 导入配置和路由
from app.core.config import settings
from app.api.endpoints import router as api_router

# 导入自定义中间件
from app.middleware.content_filter import ContentFilterMiddleware
from app.middleware.redis_limiter import RedisRequestLimiterMiddleware

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 确保静态目录存在
os.makedirs(STATIC_DIR, exist_ok=True)

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

# 重要：中间件执行顺序是从后往前，所以注册顺序很重要
# 先添加内容安全过滤中间件
app.add_middleware(ContentFilterMiddleware)

# 再添加Redis请求限制中间件
app.add_middleware(RedisRequestLimiterMiddleware, lock_timeout=300)  # 5分钟超时，防止死锁

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 创建模板引擎
templates = Jinja2Templates(directory=STATIC_DIR)

# 注册API路由
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """根路径重定向到演示页面"""
    return RedirectResponse(url="/demo")

@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    """演示页面端点"""
    return templates.TemplateResponse("demo.html", {"request": request})

@app.get("/health")
async def health():
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
logger.debug(f"静态文件目录: {STATIC_DIR}")
logger.info("已启用内容安全过滤中间件和Redis请求限制中间件") 