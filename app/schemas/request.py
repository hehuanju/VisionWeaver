#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
请求数据模型
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ImageGenerationRequest(BaseModel):
    """图像生成请求模型"""
    
    prompt: str = Field(..., description="用户的图像描述文本")
    style: Optional[str] = Field(None, description="图像风格，如'照片真实风格'、'动漫风格'等")
    size: str = Field("1024x1024", description="图像尺寸，格式如'宽度x高度'")
    count: int = Field(1, description="生成图像数量", ge=1, le=4)
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "一只橙色的猫咪坐在窗台上，窗外是蓝天白云",
                "style": "照片真实风格",
                "size": "1024x1024",
                "count": 1
            }
        } 
