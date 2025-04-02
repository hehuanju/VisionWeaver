#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像合成工具

将logo、二维码等元素合成到生成的图像中
支持位置、大小、透明度等参数调整
"""

import os
import uuid
import time
from typing import Dict, List, Optional, Union, Literal
from enum import Enum
import asyncio
from datetime import datetime
from loguru import logger

from PIL import Image, ImageEnhance, ImageFilter
from langchain_core.tools import tool

from app.tools.oss_uploader import oss_uploader


class CompositionPosition(str, Enum):
    """图像合成位置枚举"""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    CUSTOM = "custom"


class ImageComposer:
    """图像合成工具，用于将logo、二维码等元素合成到图像中"""
    
    def __init__(self):
        """初始化图像合成工具"""
        # 设置输出目录
        self.output_dir = "composed_images"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("图像合成工具初始化完成")
    
    async def compose_images(
        self,
        base_image_path: str,
        overlay_image_path: str,
        position: Union[CompositionPosition, str] = CompositionPosition.BOTTOM_RIGHT,
        overlay_size: Optional[float] = 0.2,  # 叠加图像相对于基础图像的大小比例
        margin: int = 20,  # 边距像素
        opacity: float = 1.0,  # 不透明度
        custom_position: Optional[tuple] = None,  # 自定义位置 (x, y)
        auto_upload_to_oss: bool = True  # 是否自动上传到OSS
    ) -> Dict:
        """
        将叠加图像(如logo、二维码)合成到基础图像上
        
        Args:
            base_image_path: 基础图像路径
            overlay_image_path: 叠加图像路径(如logo、二维码)
            position: 叠加位置，可选值为top_left, top_right, bottom_left, bottom_right, center, custom
            overlay_size: 叠加图像的大小比例，相对于基础图像的百分比(0.1表示10%)
            margin: 边距像素数
            opacity: 不透明度(0.0-1.0)
            custom_position: 自定义位置，仅当position为custom时有效
            auto_upload_to_oss: 是否自动上传到OSS
            
        Returns:
            包含合成结果的字典
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(base_image_path):
                logger.error(f"基础图像不存在: {base_image_path}")
                return {
                    "success": False,
                    "error": f"基础图像不存在: {base_image_path}"
                }
                
            if not os.path.exists(overlay_image_path):
                logger.error(f"叠加图像不存在: {overlay_image_path}")
                return {
                    "success": False,
                    "error": f"叠加图像不存在: {overlay_image_path}"
                }
            
            # 打开图像
            try:
                base_img = Image.open(base_image_path).convert("RGBA")
                overlay_img = Image.open(overlay_image_path).convert("RGBA")
            except Exception as e:
                logger.error(f"打开图像失败: {str(e)}")
                return {
                    "success": False,
                    "error": f"打开图像失败: {str(e)}"
                }
            
            # 调整叠加图像大小
            base_width, base_height = base_img.size
            overlay_width, overlay_height = overlay_img.size
            
            # 计算新的叠加图像尺寸
            if overlay_size:
                new_width = int(base_width * overlay_size)
                # 保持宽高比
                new_height = int(overlay_height * (new_width / overlay_width))
                overlay_img = overlay_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 调整不透明度
            if opacity < 1.0:
                # 拆分通道，只对alpha通道进行调整
                r, g, b, a = overlay_img.split()
                a = a.point(lambda i: i * opacity)
                overlay_img = Image.merge('RGBA', (r, g, b, a))
            
            # 确定位置
            position_str = position
            if isinstance(position, CompositionPosition):
                position_str = position.value
                
            # 获取叠加图像新尺寸
            overlay_width, overlay_height = overlay_img.size
            
            # 计算位置坐标
            if position_str == CompositionPosition.TOP_LEFT.value:
                position_xy = (margin, margin)
            elif position_str == CompositionPosition.TOP_RIGHT.value:
                position_xy = (base_width - overlay_width - margin, margin)
            elif position_str == CompositionPosition.BOTTOM_LEFT.value:
                position_xy = (margin, base_height - overlay_height - margin)
            elif position_str == CompositionPosition.BOTTOM_RIGHT.value:
                position_xy = (base_width - overlay_width - margin, base_height - overlay_height - margin)
            elif position_str == CompositionPosition.CENTER.value:
                position_xy = ((base_width - overlay_width) // 2, (base_height - overlay_height) // 2)
            elif position_str == CompositionPosition.CUSTOM.value and custom_position:
                position_xy = custom_position
            else:
                # 默认右下角
                position_xy = (base_width - overlay_width - margin, base_height - overlay_height - margin)
            
            # 创建新图像并粘贴基础图像
            result_img = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
            result_img.paste(base_img, (0, 0))
            
            # 粘贴叠加图像
            result_img.paste(overlay_img, position_xy, overlay_img)
            
            # 保存结果
            timestamp = int(time.time())
            base_filename = os.path.basename(base_image_path)
            overlay_filename = os.path.basename(overlay_image_path)
            result_filename = f"{self.output_dir}/composed_{timestamp}_{base_filename}"
            
            # 保存前转换为RGB模式（如果需要）
            if result_img.mode == 'RGBA':
                rgb_img = Image.new('RGB', result_img.size, (255, 255, 255))
                rgb_img.paste(result_img, mask=result_img.split()[3])  # 使用alpha通道作为蒙版
                rgb_img.save(result_filename, 'JPEG', quality=95)
            else:
                result_img.save(result_filename)
            
            logger.info(f"图像合成完成，保存到: {result_filename}")
            
            result = {
                "success": True,
                "local_path": result_filename,
                "base_image": os.path.basename(base_image_path),
                "overlay_image": os.path.basename(overlay_image_path),
                "position": position_str,
                "timestamp": timestamp
            }
            
            # 如果需要上传到OSS
            if auto_upload_to_oss:
                try:
                    logger.info("正在上传合成图像到OSS...")
                    folder_name = f"composed_{datetime.now().strftime('%Y%m%d')}"
                    oss_result = await oss_uploader.upload_image(result_filename, folder_name)
                    
                    if oss_result.get("success"):
                        result["oss_url"] = oss_result.get("url")
                        result["oss_path"] = oss_result.get("oss_path")
                        logger.info(f"合成图像上传到OSS成功: {oss_result.get('url')}")
                    else:
                        logger.warning(f"合成图像上传到OSS失败: {oss_result.get('error', '未知错误')}")
                except Exception as e:
                    logger.error(f"上传合成图像到OSS过程中出错: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"图像合成过程中出错: {str(e)}")
            logger.exception("详细错误信息:")
            return {
                "success": False,
                "error": f"图像合成失败: {str(e)}"
            }
    
    async def add_watermark(
        self,
        image_path: str,
        watermark_text: str,
        position: Union[CompositionPosition, str] = CompositionPosition.BOTTOM_RIGHT,
        font_size: int = 20,
        font_color: tuple = (255, 255, 255, 128),  # RGBA
        margin: int = 10,
        auto_upload_to_oss: bool = True
    ) -> Dict:
        """
        为图像添加文字水印
        
        Args:
            image_path: 图像路径
            watermark_text: 水印文字
            position: 水印位置
            font_size: 字体大小
            font_color: 字体颜色 (R,G,B,A)
            margin: 边距
            auto_upload_to_oss: 是否上传到OSS
            
        Returns:
            包含处理结果的字典
        """
        try:
            # 这里实现文字水印功能
            # 为简化实现，此处仅返回一个模拟成功的结果
            # 实际实现需要使用PIL的ImageDraw模块添加文字
            
            logger.warning("文字水印功能尚未完全实现，返回模拟结果")
            
            return {
                "success": True,
                "local_path": image_path,
                "watermark_text": watermark_text,
                "note": "文字水印功能尚未完全实现"
            }
            
        except Exception as e:
            logger.error(f"添加文字水印过程中出错: {str(e)}")
            return {
                "success": False,
                "error": f"添加文字水印失败: {str(e)}"
            }


# 创建全局实例
image_composer = ImageComposer()


@tool
async def compose_image(
    base_image_path: str,
    overlay_image_path: str,
    position: str = "bottom_right",
    overlay_size: Optional[float] = 0.2,
    margin: int = 20,
    opacity: float = 1.0,
    return_oss_url: bool = True
) -> Dict:
    """将logo、二维码等图像合成到基础图像中
    
    此工具用于将logo、二维码或其他图像元素合成到基础图像上，可指定位置、大小和透明度等参数。
    
    Args:
        base_image_path: 基础图像的完整文件路径，合成操作将基于此图像进行
        overlay_image_path: 要合成的图像(如logo、二维码)的完整文件路径
        position: 合成位置，可选值: top_left, top_right, bottom_left, bottom_right, center
        overlay_size: 合成图像相对于基础图像的大小比例(0.1表示10%)
        margin: 与边缘的距离(像素)
        opacity: 不透明度(0.0-1.0)
        return_oss_url: 是否返回OSS URL而非本地路径
        
    Returns:
        包含合成结果的字典，成功时包含本地路径和/或OSS URL
    """
    # 输入验证
    if not base_image_path:
        return {
            "错误": "请提供基础图像路径"
        }
        
    if not overlay_image_path:
        return {
            "错误": "请提供要合成的图像路径"
        }
    
    # 验证位置参数
    valid_positions = [p.value for p in CompositionPosition]
    if position not in valid_positions:
        position = "bottom_right"  # 默认位置
    
    # 验证大小和透明度
    if overlay_size is not None and (overlay_size <= 0 or overlay_size > 1.0):
        overlay_size = 0.2  # 默认20%
        
    if opacity < 0 or opacity > 1.0:
        opacity = 1.0  # 默认完全不透明
    
    try:
        # 调用合成方法
        result = await image_composer.compose_images(
            base_image_path=base_image_path,
            overlay_image_path=overlay_image_path,
            position=position,
            overlay_size=overlay_size,
            margin=margin,
            opacity=opacity,
            auto_upload_to_oss=return_oss_url
        )
        
        # 检查结果
        if not result.get("success"):
            return {
                "错误": result.get("error", "图像合成失败")
            }
            
        # 构造返回结果
        response = {
            "状态": "成功",
            "操作": "图像合成"
        }
        
        # 添加图片地址信息
        if return_oss_url and "oss_url" in result:
            response["图片URL"] = result["oss_url"]
            response["存储位置"] = "阿里云OSS"
        else:
            response["本地路径"] = result["local_path"]
            response["存储位置"] = "本地文件"
            
        # 添加合成信息
        response["合成位置"] = position
        response["合成尺寸比例"] = f"{overlay_size*100:.1f}%"
        
        return response
            
    except Exception as e:
        logger.error(f"处理图像合成请求时出错: {str(e)}")
        return {
            "错误": f"图像合成请求处理失败: {str(e)}"
        }


@tool
async def add_image_watermark(
    image_path: str,
    watermark_image_path: str,
    position: str = "bottom_right",
    size: float = 0.1,
    opacity: float = 0.7,
    return_oss_url: bool = True
) -> Dict:
    """为图像添加图片水印(如logo)
    
    此工具是compose_image的简化版本，专用于添加半透明水印。
    
    Args:
        image_path: 要添加水印的图像路径
        watermark_image_path: 水印图像路径(通常是logo)
        position: 水印位置，可选值: top_left, top_right, bottom_left, bottom_right
        size: 水印大小比例(相对于原图)
        opacity: 水印不透明度(0.0-1.0)
        return_oss_url: 是否返回OSS URL
        
    Returns:
        包含水印添加结果的字典
    """
    # 此工具实际上是调用compose_image的封装，使用特定参数
    return await compose_image(
        base_image_path=image_path,
        overlay_image_path=watermark_image_path,
        position=position,
        overlay_size=size,
        opacity=opacity,
        return_oss_url=return_oss_url
    ) 