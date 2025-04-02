#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OSS上传工具

将图片上传至阿里云对象存储的工具
"""

import os
import time
import asyncio
import mimetypes
from typing import Dict, Optional, Union, List
from loguru import logger

from langchain_core.tools import tool
from app.utils.aliyun_oss import oss_client

class OssUploader:
    """阿里云OSS上传工具，专用于图片上传"""
    
    def __init__(self):
        """初始化OSS上传工具"""
        self.oss_client = oss_client
        self.image_folder = "images"  # OSS中的图片存储目录
        logger.info("OSS上传工具初始化完成")
    
    async def upload_image(self, image_path: str, custom_path: Optional[str] = None) -> Dict:
        """
        上传图片到OSS
        
        Args:
            image_path: 本地图片路径
            custom_path: 自定义OSS路径，不包含基础目录
            
        Returns:
            包含上传结果的字典
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"要上传的图片不存在: {image_path}")
                return {
                    "success": False,
                    "error": f"图片文件不存在: {image_path}"
                }
            
            # 获取文件类型
            content_type = mimetypes.guess_type(image_path)[0]
            if not content_type or not content_type.startswith('image/'):
                content_type = 'image/png'  # 默认图片类型
            
            # 设置文件头
            headers = {
                'Content-Type': content_type,
                'x-oss-object-acl': 'public-read'  # 设置为公共可读
            }
            
            # 构建OSS路径
            filename = os.path.basename(image_path)
            timestamp = int(time.time())
            
            if custom_path:
                # 使用自定义路径
                oss_path = f"{self.image_folder}/{custom_path}/{filename}"
            else:
                # 使用时间戳路径
                oss_path = f"{self.image_folder}/{timestamp}/{filename}"
            
            # 异步上传文件
            url = await asyncio.to_thread(
                self.oss_client.upload_file,
                image_path,
                oss_path,
                headers
            )
            
            logger.info(f"图片上传成功: {image_path} -> {url}")
            
            return {
                "success": True,
                "url": url,
                "oss_path": oss_path,
                "content_type": content_type,
                "local_path": image_path
            }
            
        except Exception as e:
            logger.error(f"图片上传失败: {str(e)}")
            logger.exception("详细错误信息:")
            
            return {
                "success": False,
                "error": f"图片上传失败: {str(e)}",
                "local_path": image_path
            }
    
    async def batch_upload_images(self, image_paths: List[str], folder_name: Optional[str] = None) -> Dict:
        """
        批量上传图片到OSS
        
        Args:
            image_paths: 本地图片路径列表
            folder_name: 可选的文件夹名称
            
        Returns:
            包含上传结果的字典
        """
        results = []
        success_count = 0
        failed_count = 0
        
        # 创建自定义路径
        custom_path = folder_name if folder_name else f"batch_{int(time.time())}"
        
        for image_path in image_paths:
            result = await self.upload_image(image_path, custom_path)
            results.append(result)
            
            if result.get("success"):
                success_count += 1
            else:
                failed_count += 1
        
        return {
            "success": failed_count == 0,
            "total": len(image_paths),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results
        }

# 创建全局实例
oss_uploader = OssUploader()

@tool
async def upload_image_to_oss(image_path: str, folder_name: Optional[str] = None) -> Dict:
    """将图片上传到阿里云OSS对象存储服务
    
    将本地生成的图片上传到云存储，获取可公开访问的URL。适用于需要分享图片或将图片嵌入到网页、文档中的场景。
    
    Args:
        image_path: 本地图片文件路径，必须是完整的文件路径
        folder_name: 可选，上传到OSS的子文件夹名称，用于组织管理图片
        
    Returns:
        包含上传结果的字典，成功时包含图片URL、OSS路径等信息
    """
    if not image_path:
        return {
            "错误": "请提供图片路径"
        }
    
    # 确保路径存在
    if not os.path.exists(image_path):
        return {
            "错误": f"文件不存在: {image_path}"
        }
    
    # 上传图片
    logger.info(f"开始上传图片到OSS: {image_path}")
    result = await oss_uploader.upload_image(image_path, folder_name)
    
    if result.get("success"):
        logger.info(f"图片上传成功: {result.get('url')}")
        return {
            "状态": "成功",
            "图片URL": result.get("url"),
            "OSS路径": result.get("oss_path")
        }
    else:
        logger.error(f"图片上传失败: {result.get('error')}")
        return {
            "状态": "失败",
            "错误": result.get("error", "未知错误")
        } 