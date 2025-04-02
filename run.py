#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文生图Agent启动脚本
"""

import os
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    # 启动FastAPI应用
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("APP_ENV") == "development"
    ) 
