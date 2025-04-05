#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
响应数据模型
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime


class ImageGenerationResponse(BaseModel):
    """图像生成响应模型"""
    
    status: str = Field(..., description="请求状态，如'processing'、'completed'、'failed'")
    message: str = Field(..., description="响应消息")
    request_id: str = Field(..., description="请求ID，可用于查询生成状态")
    estimated_time: Optional[int] = Field(None, description="预计处理时间（秒）")
    created_at: datetime = Field(default_factory=datetime.now, description="响应创建时间")
    
    # 任务完成后的返回值
    images: Optional[List[str]] = Field(None, description="生成图片的URL列表")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "completed",
                "message": "图像生成成功",
                "request_id": "gen_123456789",
                "estimated_time": None,
                "created_at": "2025-03-31T08:45:30.123456",
                "images": [
                    "https://vision-weaver.oss-cn-hangzhou.aliyuncs.com/images/gen_123456789_1.jpg"
                ]
            }
        }


class GenerationStatus(BaseModel):
    """生成状态查询响应"""
    
    status: str = Field(..., description="任务状态")
    progress: Optional[float] = Field(None, description="生成进度，0-100")
    message: str = Field(..., description="状态描述") 
