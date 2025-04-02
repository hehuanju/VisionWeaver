#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
应用配置模块
"""

import os
from typing import Any, Dict, List, Optional, Union
from pydantic import validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    PROJECT_NAME: str = "VisionWeaver"
    API_V1_STR: str = "/v1"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # API配置
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # 大模型配置
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_BASE: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    AGENT_MODEL: str = os.getenv("AGENT_MODEL", "anthropic/claude-3.5-haiku")
    USE_OPENROUTER: bool = os.getenv("USE_OPENROUTER", "true").lower() == "true"
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_API_BASE: Optional[str] = os.getenv("DEEPSEEK_API_BASE")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    OPENWEBUI_API_KEY: Optional[str] = os.getenv("OPENWEBUI_API_KEY")
    AGENT_MODEL: str = os.getenv("AGENT_MODEL", "gemini-1.5-pro")
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "SQLALCHEMY_DATABASE_URI", 
        "sqlite+aiosqlite:///./app/db/vision_weaver.db"
    )
    
    # Redis配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "127.0.0.1")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # 阿里云OSS配置
    OSS_ACCESS_KEY: Optional[str] = os.getenv("OSS_ACCESS_KEY")
    OSS_SECRET_KEY: Optional[str] = os.getenv("OSS_SECRET_KEY")
    OSS_BUCKET: Optional[str] = os.getenv("OSS_BUCKET")
    OSS_ENDPOINT: Optional[str] = os.getenv("OSS_ENDPOINT")
    OSS_REGION: Optional[str] = os.getenv("OSS_REGION")
    AUTO_UPLOAD_TO_OSS: bool = os.getenv("AUTO_UPLOAD_TO_OSS", "false").lower() == "true"
    
    # CORS配置
    BACKEND_CORS_ORIGINS: List[str] = []
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # 日志设置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")
    LOG_RETENTION: str = os.getenv("LOG_RETENTION", "7 days")
    LOG_ROTATION: str = os.getenv("LOG_ROTATION", "00:00")
    LOG_COMPRESSION: str = os.getenv("LOG_COMPRESSION", "zip")
    
    # 水印设置
    WATERMARK: str = os.getenv("WATERMARK", "VISIONWEAVER")
    
    # 服务器设置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 创建全局设置实例
settings = Settings() 
