#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片生成机器人工具

基于提示词生成图片的机器人，使用Gemini文生图模型
使用Google AI官方API接口调用Gemini图像生成模型
"""

import os
import json
import base64
import aiohttp
import asyncio
import re
from typing import Dict, List, Optional, Any, Union
from loguru import logger
from urllib.parse import urlparse

from langchain_core.tools import tool
from langchain_core.tools import ToolException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.tools.oss_uploader import oss_uploader

class ImageGeneratorBot:
    """图片生成机器人，使用Gemini模型通过Google API实现"""
    
    def __init__(self):
        """初始化图片生成机器人"""
        # 设置API密钥
        self.api_key = settings.GOOGLE_API_KEY
        
        # 设置是否自动上传图片到OSS
        self.auto_upload_to_oss = settings.AUTO_UPLOAD_TO_OSS if hasattr(settings, 'AUTO_UPLOAD_TO_OSS') else False
        
        if not self.api_key:
            logger.warning("未配置Google API密钥，图片生成机器人不可用")
            self.is_configured = False
        else:
            self.is_configured = True
            
            # 设置模型ID - 使用谷歌官方支持图像生成的模型
            self.model_id = "gemini-2.0-flash-exp-image-generation"
            logger.info("图片生成机器人初始化完成")
            logger.info(f"使用Google AI API，模型: {self.model_id}")
            
            if self.auto_upload_to_oss:
                logger.info("已启用自动上传图片到OSS功能")
            else:
                logger.info("未启用自动上传图片到OSS功能")
    
    async def generate_image(self, prompt: str, size: Optional[str] = "1024x1024") -> Dict:
        """
        根据提示词生成图片
        
        Args:
            prompt: 图片生成提示词
            size: 生成图片的尺寸，格式为"宽x高"，如"1024x1024"、"512x512"、"768x768"等
            
        Returns:
            包含图片URL或Base64数据的字典，或者错误信息
        """
        if not self.is_configured:
            raise ToolException("图片生成机器人未正确配置，请检查API密钥")
        
        if not prompt or len(prompt.strip()) < 5:
            raise ToolException("提示词过短，请提供更详细的描述")
        
        logger.info(f"开始生成图片，提示词: {prompt[:50]}...")
        
        try:
            # 导入Google的生成式AI库
            import google.generativeai as genai
            
            # 配置API密钥
            genai.configure(api_key=self.api_key)
            
            # 构建提示，添加尺寸信息到提示词中
            # 从尺寸参数中提取宽和高
            try:
                width, height = map(int, size.split("x"))
                aspect_ratio = width / height
                
                # 检查提示词中是否已经包含了比例信息
                ratio_match = re.search(r'按照(\d+:\d+)的比例生成图片', prompt)
                if ratio_match:
                    # 用户已经指定了比例，使用用户指定的比例描述
                    user_ratio = ratio_match.group(1)
                    logger.info(f"检测到用户在提示词中指定的图片比例: {user_ratio}")
                    size_desc = f"按照{user_ratio}比例的"
                elif aspect_ratio == 1:
                    size_desc = "正方形(1:1比例)"
                elif aspect_ratio > 1:
                    size_desc = f"横向矩形({width}x{height}尺寸，宽高比约{aspect_ratio:.1f}:1)"
                else:
                    size_desc = f"纵向矩形({width}x{height}尺寸，宽高比约1:{1/aspect_ratio:.1f})"
            except:
                size_desc = f"{size}尺寸"
                
            # 构建增强后的提示词，保留用户原始的比例要求
            if ratio_match:
                enhanced_prompt = f"请根据以下描述生成一幅高质量{size_desc}图片: {prompt}"
            else:
                enhanced_prompt = f"请根据以下描述生成一幅高质量{size_desc}图片: {prompt}。确保图片尺寸为{size}。生成图片和简短描述。"
            
            # 发送请求
            logger.info("使用Google Generative AI原生API发送请求...")
            logger.info(f"增强后的提示词: {enhanced_prompt[:100]}...")
            
            # 获取支持图像生成的模型
            model = genai.GenerativeModel(self.model_id)
            
            # 使用正确的配置参数，使用正确的枚举值格式 (全大写)
            # 根据Gemini API文档，尺寸参数不通过generation_config参数传递
            response = await asyncio.to_thread(
                model.generate_content,
                enhanced_prompt,
                generation_config={
                    "temperature": 0.7,
                    "response_modalities": ["TEXT", "IMAGE"]  # 使用全大写的枚举值
                }
            )
            
            # 打印详细的响应信息用于调试
            logger.debug(f"Gemini API响应类型: {type(response)}")
            logger.debug(f"Gemini API响应属性: {dir(response)}")
            
            # 检查响应结构
            if hasattr(response, 'parts'):
                logger.debug(f"响应包含 {len(response.parts)} 个部分")
                
                # 查找包含图像的部分
                for i, part in enumerate(response.parts):
                    logger.debug(f"部分 {i+1} 类型: {type(part)}")
                    
                    # 如果部分包含内联数据(图像)
                    if hasattr(part, 'inline_data') and hasattr(part.inline_data, 'mime_type') and part.inline_data.mime_type.startswith('image/'):
                        logger.info(f"从响应中获得直接图像数据，MIME类型: {part.inline_data.mime_type}")
                        
                        # 获取图像数据
                        b64_data = part.inline_data.data
                        logger.debug(f"获取到图像数据，长度: {len(b64_data) if b64_data else 0}")
                        
                        # 检查数据的有效性
                        if b64_data:
                            # 检查数据类型
                            logger.debug(f"图像数据类型: {type(b64_data).__name__}")
                            
                            # 如果是字节数据，转换为Base64字符串
                            if isinstance(b64_data, bytes):
                                try:
                                    # 检查是否为原始图像数据（而非Base64编码）
                                    is_png = b64_data.startswith(b'\x89PNG\r\n\x1a\n')
                                    is_jpeg = b64_data.startswith(b'\xff\xd8\xff')
                                    
                                    if is_png or is_jpeg:
                                        logger.info("收到的是原始二进制图像数据，正在转换为Base64编码...")
                                        b64_data = base64.b64encode(b64_data).decode('utf-8')
                                    else:
                                        # 尝试作为UTF-8解码
                                        logger.info("尝试将字节数据解码为UTF-8字符串...")
                                        b64_data = b64_data.decode('utf-8')
                                except Exception as e:
                                    logger.warning(f"处理字节数据时出错: {str(e)}，强制转换为Base64...")
                                    b64_data = base64.b64encode(b64_data).decode('utf-8')
                                    
                            # 检查Base64数据的有效性
                            if not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in b64_data):
                                logger.warning("API返回的Base64数据包含无效字符!")
                            
                            # 记录数据特征
                            logger.debug(f"图像数据前20个字符: {b64_data[:20] if len(b64_data) >= 20 else b64_data}")
                            logger.debug(f"图像数据后20个字符: {b64_data[-20:] if len(b64_data) >= 20 else b64_data}")
                            
                            # 尝试解码小部分数据，测试有效性
                            try:
                                test_chunk = b64_data[:100] if len(b64_data) > 100 else b64_data
                                test_decode = base64.b64decode(test_chunk)
                                logger.debug(f"测试解码成功，解码后长度: {len(test_decode)}")
                            except Exception as e:
                                logger.warning(f"Base64测试解码失败: {str(e)}")
                        
                        # 返回图像数据
                        return {
                            "url": None,
                            "b64_json": b64_data,
                            "generation_info": {
                                "prompt": prompt,
                                "model": self.model_id,
                                "source": "google_genai_image_generation",
                                "size": size
                            }
                        }
                    
                    # 如果部分包含文本，检查是否有图像的URL
                    elif hasattr(part, 'text'):
                        text = part.text
                        logger.debug(f"部分 {i+1} 文本内容: {text[:200]}...")
                        
                        # 尝试提取URL
                        image_url = self._extract_url_from_text(text)
                        if image_url and not any(domain in image_url for domain in ["lexica.art", "image.lexica.art"]):
                            logger.info(f"从响应文本中提取到图片URL: {image_url}")
                            return {
                                "url": image_url,
                                "b64_json": None,
                                "generation_info": {
                                    "prompt": prompt,
                                    "model": self.model_id,
                                    "source": "google_genai_url_extraction",
                                    "size": size
                                }
                            }
                        
                        # 尝试提取base64数据
                        base64_data = self._extract_base64_from_text(text)
                        if base64_data:
                            logger.info(f"从响应文本中提取到base64图片数据，长度: {len(base64_data)}")
                            return {
                                "url": None,
                                "b64_json": base64_data,
                                "generation_info": {
                                    "prompt": prompt,
                                    "model": self.model_id,
                                    "source": "google_genai_text_base64",
                                    "size": size
                                }
                            }
            
            # 没有找到有效的图像数据
            logger.error("无法从Gemini API响应中获取图像数据")
            return {
                "error": "无法生成有效的图片",
                "generation_info": {
                    "prompt": prompt,
                    "model": self.model_id,
                    "source": "error",
                    "size": size
                }
            }
                
        except Exception as e:
            logger.error(f"图片生成过程中发生错误: {str(e)}")
            logger.exception("详细错误信息:")
            return {
                "error": f"图片生成失败: {str(e)}",
                "generation_info": {
                    "prompt": prompt,
                    "model": self.model_id,
                    "source": "exception",
                    "size": size
                }
            }
    
    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取JSON部分"""
        import re
        
        # 尝试不同的JSON提取模式
        patterns = [
            r'```json\s*(.*?)\s*```',  # Markdown 代码块中的JSON
            r'```\s*(.*?)\s*```',      # 任何代码块
            r'{.*}',                    # 大括号包围的任何内容
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0]
        
        return None
    
    def _extract_url_from_text(self, text: str) -> Optional[str]:
        """从文本中提取图片URL"""
        import re
        
        # 常见图片URL模式
        url_patterns = [
            r'https?://\S+\.(?:jpg|jpeg|png|webp|gif)\b',  # 直接图片链接
            r'https?://\S+/\S+\.(?:jpg|jpeg|png|webp|gif)\b',  # 路径中包含的图片链接
            r'https?://\S+\.(com|org|net|io|ai)/\S+\.(jpg|jpeg|png|webp|gif)',  # 域名+图片扩展名
            r'https?://\S+\.(com|org|net|io|ai)/\S+',  # 一般URL
            r'https?://\S+'  # 任何URL
        ]
        
        for pattern in url_patterns:
            urls = re.findall(pattern, text)
            if urls:
                # 返回第一个匹配的URL
                for url in urls:
                    if isinstance(url, tuple):
                        # 如果正则表达式有捕获组，结果会是元组
                        full_url = text[text.find('http'):text.find(url[-1]) + len(url[-1])]
                        return full_url
                    return url
        
        return None
    
    def _extract_base64_from_data_url(self, data_url: str) -> Optional[str]:
        """从data URL中提取base64编码的数据部分"""
        if not data_url.startswith('data:image'):
            return None
            
        # 找到base64数据部分开始的位置
        base64_start = data_url.find('base64,')
        if base64_start < 0:
            return None
            
        # 提取base64部分
        base64_start += 7  # 跳过'base64,'
        return data_url[base64_start:]
    
    def _extract_base64_from_text(self, text: str) -> Optional[str]:
        """从文本中提取base64编码数据"""
        # 首先尝试查找data URL
        data_url_match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', text)
        if data_url_match:
            return data_url_match.group(1)
            
        # 尝试查找纯base64字符串(长度至少100且符合base64字符)
        base64_match = re.search(r'[A-Za-z0-9+/=]{100,}', text)
        if base64_match:
            return base64_match.group(0)
            
        return None
    
    async def save_image(self, image_data: Dict, output_dir: str = "generated_images", force_upload_to_oss: Optional[bool] = False) -> str:
        """
        保存生成的图片到本地
        
        Args:
            image_data: 包含图片URL或Base64数据的字典
            output_dir: 输出目录
            force_upload_to_oss: 是否强制上传到OSS
            
        Returns:
            保存的图片路径
        """
        if not image_data:
            logger.error("保存图片失败: 图片数据为空")
            return None
            
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名 (时间戳)
            timestamp = int(asyncio.get_event_loop().time())
            filename = f"{output_dir}/image_{timestamp}.png"
            
            # 如果有Base64数据，直接解码保存
            if image_data.get("b64_json"):
                try:
                    # 记录原始Base64数据的部分特征
                    b64_data = image_data["b64_json"]
                    logger.debug(f"Base64数据长度: {len(b64_data)}")
                    logger.debug(f"Base64数据前20个字符: {b64_data[:20] if len(b64_data) > 20 else b64_data}")
                    logger.debug(f"Base64数据后20个字符: {b64_data[-20:] if len(b64_data) > 20 else b64_data}")
                    
                    # 检查是否为有效的Base64字符串
                    if not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in b64_data):
                        logger.warning("Base64数据包含无效字符！")
                        
                        # 清理Base64字符串，去除非法字符
                        cleaned_b64 = ''.join(c for c in b64_data if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
                        logger.info(f"清理后的Base64数据长度: {len(cleaned_b64)}")
                        
                        if len(cleaned_b64) != len(b64_data):
                            logger.warning(f"清理了 {len(b64_data) - len(cleaned_b64)} 个无效字符")
                            b64_data = cleaned_b64
                    
                    # 如果Base64数据以data:开头，则需要提取出实际的Base64部分
                    if b64_data.startswith('data:'):
                        logger.debug("Base64数据包含data URL前缀，正在提取...")
                        b64_data = self._extract_base64_from_data_url(b64_data)
                        logger.debug(f"提取后的Base64数据长度: {len(b64_data) if b64_data else 0}")
                    
                    if not b64_data:
                        logger.error("无法提取有效的Base64数据")
                        return None
                    
                    # 解码Base64数据
                    img_bytes = base64.b64decode(b64_data)
                    logger.debug(f"解码后的二进制数据长度: {len(img_bytes)}")
                    
                    # 检查解码后的数据是否为有效的图片格式（检查PNG或JPEG文件头）
                    is_png = img_bytes.startswith(b'\x89PNG\r\n\x1a\n')
                    is_jpeg = img_bytes.startswith(b'\xff\xd8\xff')
                    logger.debug(f"解码数据是否为PNG: {is_png}, 是否为JPEG: {is_jpeg}")
                    
                    if not (is_png or is_jpeg):
                        logger.warning("解码后的数据不是标准图片格式！")
                    
                    # 写入文件
                    with open(filename, "wb") as f:
                        f.write(img_bytes)
                    
                    # 检查写入后的文件大小
                    file_size = os.path.getsize(filename)
                    logger.info(f"已保存生成的图片到: {filename} (文件大小: {file_size} 字节)")
                    
                    if file_size < 100:
                        logger.error(f"警告：保存的文件过小 ({file_size} 字节)，可能已损坏!")
                        return None
                    
                    # 判断是否需要上传到OSS
                    should_upload_to_oss = self.auto_upload_to_oss or force_upload_to_oss
                    
                    # 如果启用了自动上传到OSS且文件大小正常
                    if should_upload_to_oss and file_size >= 100:
                        logger.info("开始上传图片到OSS...")
                        prompt_text = image_data.get("generation_info", {}).get("prompt", "")
                        # 从提示词中提取前10个字符作为文件夹名称
                        folder_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', prompt_text[:10])
                        
                        try:
                            oss_result = await oss_uploader.upload_image(filename, folder_name)
                            if oss_result.get("success"):
                                logger.info(f"图片已成功上传到OSS: {oss_result.get('url')}")
                                # 将OSS URL添加到图像数据中
                                image_data["oss_url"] = oss_result.get("url")
                                image_data["oss_path"] = oss_result.get("oss_path")
                            else:
                                logger.error(f"上传到OSS失败: {oss_result.get('error')}")
                                if force_upload_to_oss:
                                    logger.error("由于要求强制上传到OSS但失败，返回错误")
                                    return None
                        except Exception as e:
                            logger.error(f"上传到OSS过程中发生错误: {str(e)}")
                            if force_upload_to_oss:
                                logger.error("由于要求强制上传到OSS但失败，返回错误")
                                return None
                    elif force_upload_to_oss:
                        logger.error("要求强制上传到OSS但未配置OSS或图片文件异常")
                        return None
                    
                    return filename
                except Exception as e:
                    logger.error(f"保存Base64图片数据时发生错误: {str(e)}")
                    logger.exception("详细错误信息:")
                    return None
                
            # 如果有URL，下载并保存
            elif image_data.get("url"):
                # 检查是否为data URL
                if image_data["url"].startswith('data:image'):
                    b64_data = self._extract_base64_from_data_url(image_data["url"])
                    if b64_data:
                        with open(filename, "wb") as f:
                            img_bytes = base64.b64decode(b64_data)
                            f.write(img_bytes)
                        logger.info(f"已保存base64数据URL图片到: {filename}")
                        
                        # 判断是否需要上传到OSS
                        should_upload_to_oss = self.auto_upload_to_oss or force_upload_to_oss
                        
                        # 如果启用了自动上传到OSS
                        if should_upload_to_oss:
                            logger.info("开始上传图片到OSS...")
                            prompt_text = image_data.get("generation_info", {}).get("prompt", "")
                            folder_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', prompt_text[:10])
                            
                            try:
                                oss_result = await oss_uploader.upload_image(filename, folder_name)
                                if oss_result.get("success"):
                                    logger.info(f"图片已成功上传到OSS: {oss_result.get('url')}")
                                    image_data["oss_url"] = oss_result.get("url")
                                    image_data["oss_path"] = oss_result.get("oss_path")
                                else:
                                    logger.error(f"上传到OSS失败: {oss_result.get('error')}")
                                    if force_upload_to_oss:
                                        return None
                            except Exception as e:
                                logger.error(f"上传到OSS过程中发生错误: {str(e)}")
                                if force_upload_to_oss:
                                    return None
                        elif force_upload_to_oss:
                            logger.error("要求强制上传到OSS但未配置OSS")
                            return None
                        
                        return filename
                # 普通URL
                else:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_data["url"]) as response:
                            if response.status == 200:
                                with open(filename, "wb") as f:
                                    f.write(await response.read())
                                logger.info(f"已下载并保存图片到: {filename}")
                                
                                # 判断是否需要上传到OSS
                                should_upload_to_oss = self.auto_upload_to_oss or force_upload_to_oss
                                
                                # 如果启用了自动上传到OSS
                                if should_upload_to_oss:
                                    logger.info("开始上传图片到OSS...")
                                    prompt_text = image_data.get("generation_info", {}).get("prompt", "")
                                    folder_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', prompt_text[:10])
                                    
                                    try:
                                        oss_result = await oss_uploader.upload_image(filename, folder_name)
                                        if oss_result.get("success"):
                                            logger.info(f"图片已成功上传到OSS: {oss_result.get('url')}")
                                            image_data["oss_url"] = oss_result.get("url")
                                            image_data["oss_path"] = oss_result.get("oss_path")
                                        else:
                                            logger.error(f"上传到OSS失败: {oss_result.get('error')}")
                                            if force_upload_to_oss:
                                                return None
                                    except Exception as e:
                                        logger.error(f"上传到OSS过程中发生错误: {str(e)}")
                                        if force_upload_to_oss:
                                            return None
                                elif force_upload_to_oss:
                                    logger.error("要求强制上传到OSS但未配置OSS")
                                    return None
                                
                                return filename
                            else:
                                logger.warning(f"下载图片失败: {response.status}")
                                return None
            else:
                logger.warning("没有有效的图片数据可保存")
                return None
                
        except Exception as e:
            logger.error(f"保存图片时发生错误: {str(e)}")
            return None

    async def _resize_image_to_target(self, image_data: Dict, target_size: str) -> Dict:
        """
        将生成的图片调整为目标尺寸
        
        Args:
            image_data: 包含图片URL或Base64数据的字典
            target_size: 目标尺寸，格式为"宽x高"
            
        Returns:
            调整后的图片数据字典
        """
        if not image_data or "b64_json" not in image_data or not image_data["b64_json"]:
            logger.warning("无法调整图片尺寸：没有有效的Base64图片数据")
            return image_data
            
        try:
            # 解析目标尺寸
            width, height = map(int, target_size.split("x"))
            
            # 导入Pillow库
            from PIL import Image
            import io
            
            # 解码Base64数据为图片
            img_bytes = base64.b64decode(image_data["b64_json"])
            img = Image.open(io.BytesIO(img_bytes))
            
            # 记录原始图片尺寸
            original_size = f"{img.width}x{img.height}"
            logger.info(f"图片原始尺寸: {original_size}, 目标尺寸: {target_size}")
            
            # 检查原始图片是否已经符合用户请求的比例
            # 如果原始图片比例与用户指定的比例相近，且仅是尺寸不同，保留原始比例
            if "generation_info" in image_data and "prompt" in image_data["generation_info"]:
                prompt = image_data["generation_info"]["prompt"]
                ratio_match = re.search(r'按照(\d+):(\d+)的比例生成图片', prompt)
                
                if ratio_match:
                    user_ratio_w = int(ratio_match.group(1))
                    user_ratio_h = int(ratio_match.group(2))
                    user_aspect_ratio = user_ratio_w / user_ratio_h
                    img_aspect_ratio = img.width / img.height
                    
                    # 如果生成的图片比例已经接近用户要求的比例（允许5%误差），保留原始比例
                    if abs(img_aspect_ratio - user_aspect_ratio) / user_aspect_ratio < 0.05:
                        logger.info(f"生成的图片比例({img_aspect_ratio:.2f})接近用户指定比例({user_aspect_ratio:.2f})，保留原始比例")
                        # 根据用户指定的比例重新计算目标尺寸
                        if width >= height:
                            new_width = width
                            new_height = int(width / user_aspect_ratio)
                        else:
                            new_height = height
                            new_width = int(height * user_aspect_ratio)
                        
                        target_size = f"{new_width}x{new_height}"
                        logger.info(f"根据用户指定比例调整目标尺寸为: {target_size}")
            
            # 如果尺寸已经匹配，无需调整
            if f"{img.width}x{img.height}" == target_size:
                logger.info("图片尺寸已匹配目标尺寸，无需调整")
                return image_data
                
            # 调整图片尺寸
            resized_img = img.resize((width, height), Image.LANCZOS)
            
            # 转换回Base64
            buffer = io.BytesIO()
            resized_img.save(buffer, format="PNG")
            b64_resized = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # 更新图片数据
            image_data["b64_json"] = b64_resized
            image_data["generation_info"]["original_size"] = original_size
            image_data["generation_info"]["resized"] = True
            
            logger.info(f"已将图片从{original_size}调整为{target_size}")
            return image_data
            
        except Exception as e:
            logger.error(f"调整图片尺寸时发生错误: {str(e)}")
            logger.exception("详细错误信息:")
            return image_data  # 返回原始图片数据


# 创建全局实例
image_generator_bot = ImageGeneratorBot()


@tool
async def generate_image(
    prompt: str,
    size: Optional[str] = "1024x1024",
    return_oss_url: Optional[bool] = False
) -> Dict:
    """使用Gemini模型根据文字描述生成图片
    
    根据详细的文字描述生成高质量图片。支持多种尺寸和选项。
    
    Args:
        prompt: 详细的图片描述文本，需要描述想要生成的图片内容、风格和元素
        size: 生成图片的尺寸，格式为"宽x高"，如"1024x1024"、"512x512"、"768x768"等
        return_oss_url: 是否直接返回网络URL（自动上传OSS），默认为False
        
    Returns:
        包含图片URL或Base64数据的字典，或错误信息
    """
    # 输入验证
    if not prompt:
        return {
            "错误": "请提供图片描述"
        }
        
    # 确保输入非空
    prompt = prompt.strip()
    if len(prompt) < 5:
        return {
            "错误": "请提供更详细的图片描述，至少5个字符"
        }
    
    # 验证尺寸格式
    if not re.match(r'^\d+x\d+$', size):
        return {
            "错误": f"图片尺寸格式错误: {size}，应为如 '1024x1024' 的格式"
        }
        
    try:
        # 调用图片生成机器人生成图片
        result = await image_generator_bot.generate_image(prompt, size=size)
        
        # 检查是否有错误
        if "error" in result:
            logger.error(f"图片生成工具错误: {result['error']}")
            return {
                "错误": result["error"]
            }
        
        if not result:
            logger.error("图片生成失败: 生成结果为空")
            return {
                "错误": "图片生成失败: 未能生成有效的图片数据"
            }
        
        # 如果有Base64数据，尝试调整尺寸
        if "b64_json" in result and result["b64_json"]:
            try:
                # 尝试安装Pillow库（如果还没有安装）
                try:
                    import PIL
                except ImportError:
                    logger.warning("Pillow库未安装，无法调整图片尺寸。尝试安装...")
                    import sys
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
                    logger.info("Pillow库安装成功")
                
                # 调整图片尺寸
                result = await image_generator_bot._resize_image_to_target(result, size)
            except Exception as e:
                logger.warning(f"调整图片尺寸失败: {str(e)}，将使用原始尺寸")
        
        # 配置是否强制上传到OSS
        force_upload_to_oss = return_oss_url  
        
        # 尝试保存图片到本地
        image_path = await image_generator_bot.save_image(result, force_upload_to_oss=force_upload_to_oss)
        
        # 将本地路径添加到结果中
        if image_path:
            result["local_path"] = image_path
        
        # 构建返回结果
        response = {}
        
        # 如果请求直接返回OSS URL，且有OSS URL可用
        if return_oss_url:
            if "oss_url" in result:
                # 仅返回OSS URL
                return {
                    "图片URL": result["oss_url"]
                }
            else:
                # 如果无法获取OSS URL，返回错误
                return {
                    "错误": "无法获取OSS图片URL，请检查OSS配置或稍后重试"
                }
        else:
            # 正常返回完整信息
            # 如果有OSS URL，优先使用
            if "oss_url" in result:
                response["图片URL"] = result["oss_url"]
                response["存储位置"] = "阿里云OSS"
                response["OSS路径"] = result.get("oss_path", "")
            # 如果有普通URL，次之
            elif "url" in result:
                response["图片URL"] = result["url"]
                response["存储位置"] = "远程服务器"
            # 如果都没有但有本地路径
            elif "local_path" in result:
                response["图片URL"] = f"file://{result['local_path']}"
                response["存储位置"] = "本地文件系统"
                response["本地路径"] = result["local_path"]
            # 如果都没有但有Base64数据
            elif "b64_json" in result:
                # 截断过长的base64数据
                b64_preview = result["b64_json"][:30] + "..." if len(result["b64_json"]) > 30 else result["b64_json"]
                response["图片数据"] = f"Base64编码 ({len(result['b64_json'])} 字节，预览: {b64_preview})"
                response["存储位置"] = "内存"
                if "local_path" in result:
                    response["本地路径"] = result["local_path"]
            
            # 添加生成信息
            if "generation_info" in result:
                gen_info = result["generation_info"]
                if "model" in gen_info:
                    response["使用模型"] = gen_info["model"]
                if "source" in gen_info:
                    response["生成来源"] = gen_info["source"]
                if "size" in gen_info:
                    response["图片尺寸"] = gen_info["size"]
            
            logger.info(f"图片生成成功，即将返回结果")
            return response
            
    except ToolException as e:
        # 返回错误信息而不是直接抛出异常，这样可以在UI中显示
        logger.error(f"图片生成工具异常: {str(e)}")
        return {
            "错误": str(e)
        }
    except Exception as e:
        # 其他异常也转换为友好的错误消息
        logger.error(f"图片生成过程中发生未知错误: {str(e)}")
        return {
            "错误": f"图片生成失败: {str(e)}"
        } 
