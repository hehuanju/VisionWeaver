#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API端点路由模块
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from loguru import logger
from typing import Any, Dict, List, Optional
import os
import uuid
import asyncio
import json
from datetime import datetime
import redis.asyncio as redis

from app.core.config import settings
from app.schemas.request import ImageGenerationRequest
from app.schemas.response import ImageGenerationResponse, GenerationStatus
from app.core.engine import VisionWeaverEngine

# 创建路由器
router = APIRouter()

# 创建一个全局引擎实例
engine = VisionWeaverEngine(print_debug=False)

# Redis相关配置
REDIS_STATUS_PREFIX = "visionweaver:task_status:"
REDIS_RESULT_PREFIX = "visionweaver:task_result:"
# Redis存储的过期时间（秒）
REDIS_TASK_EXPIRY = 86400  # 24小时

# 临时文件存储目录
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Redis客户端
redis_client = None
redis_initialized = False


async def get_redis_client():
    """获取Redis客户端"""
    global redis_client, redis_initialized
    
    if not redis_initialized:
        try:
            # 从settings构建Redis URL
            auth_part = ""
            if getattr(settings, "REDIS_PASSWORD", None):
                auth_part = f":{settings.REDIS_PASSWORD}@"
            redis_url = f"redis://{auth_part}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
            
            redis_client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            redis_initialized = True
            logger.info(f"API端点已连接到Redis服务器: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"API端点连接Redis失败: {str(e)}")
            redis_initialized = False
            raise
    
    return redis_client


async def set_task_status(request_id: str, status_data: dict):
    """将任务状态存储到Redis"""
    try:
        client = await get_redis_client()
        status_key = f"{REDIS_STATUS_PREFIX}{request_id}"
        await client.set(status_key, json.dumps(status_data), ex=REDIS_TASK_EXPIRY)
        logger.debug(f"已更新Redis任务状态: {request_id}")
    except Exception as e:
        logger.error(f"存储任务状态到Redis失败: {str(e)}")


async def get_task_status(request_id: str) -> dict:
    """从Redis获取任务状态"""
    try:
        client = await get_redis_client()
        status_key = f"{REDIS_STATUS_PREFIX}{request_id}"
        status_json = await client.get(status_key)
        
        if not status_json:
            return None
            
        return json.loads(status_json)
    except Exception as e:
        logger.error(f"从Redis获取任务状态失败: {str(e)}")
        return None


async def set_task_result(request_id: str, result_data: dict):
    """将任务结果存储到Redis"""
    try:
        client = await get_redis_client()
        result_key = f"{REDIS_RESULT_PREFIX}{request_id}"
        await client.set(result_key, json.dumps(result_data), ex=REDIS_TASK_EXPIRY)
        logger.debug(f"已存储Redis任务结果: {request_id}")
    except Exception as e:
        logger.error(f"存储任务结果到Redis失败: {str(e)}")


async def get_task_result(request_id: str) -> dict:
    """从Redis获取任务结果"""
    try:
        client = await get_redis_client()
        result_key = f"{REDIS_RESULT_PREFIX}{request_id}"
        result_json = await client.get(result_key)
        
        if not result_json:
            return None
            
        return json.loads(result_json)
    except Exception as e:
        logger.error(f"从Redis获取任务结果失败: {str(e)}")
        return None


async def process_image_generation(
    request_id: str,
    prompt: str,
    image_paths: List[str] = None,
    model: str = "gemini-1.5-pro",
    temperature: float = 0.7
):
    """后台处理图像生成任务"""
    try:
        # 更新任务状态为进行中
        status_data = {
            "status": "processing",
            "progress": 0,
            "message": "正在初始化生成任务...",
            "start_time": datetime.now().isoformat()
        }
        await set_task_status(request_id, status_data)
        
        # 创建专用引擎实例
        task_engine = VisionWeaverEngine(
            model_name=model,
            temperature=temperature,
            print_debug=False
        )
        
        # 更新进度
        status_data.update({
            "progress": 10,
            "message": "正在分析需求..."
        })
        await set_task_status(request_id, status_data)
        
        # 执行图像生成
        result = await task_engine.arun(prompt, request_id, input_images=image_paths)
        
        # 更新进度
        status_data.update({
            "progress": 90,
            "message": "图像生成完成，准备返回结果..."
        })
        await set_task_status(request_id, status_data)
        
        # 提取结果信息
        images = []
        if "image_result" in result and "图片URL" in result["image_result"]:
            images.append(result["image_result"]["图片URL"])
        
        # 如果有合成图像，优先使用合成后的图像
        if "composed_image_result" in result and result["composed_image_result"]:
            if "图片URL" in result["composed_image_result"]:
                images = [result["composed_image_result"]["图片URL"]]
        
        # 保存结果
        result_data = {
            "status": "completed",
            "message": "图像生成成功",
            "images": images,
            "output": result.get("output"),
            "created_at": datetime.now().isoformat()
        }
        await set_task_result(request_id, result_data)
        
        # 更新最终状态
        status_data.update({
            "status": "completed",
            "progress": 100,
            "message": "图像生成成功"
        })
        await set_task_status(request_id, status_data)
        
        # 清理临时文件
        if image_paths:
            for path in image_paths:
                if os.path.exists(path) and path.startswith(TEMP_UPLOAD_DIR):
                    try:
                        os.remove(path)
                        logger.debug(f"删除临时文件: {path}")
                    except Exception as e:
                        logger.error(f"删除临时文件失败: {str(e)}")
        
    except Exception as e:
        logger.exception(f"图像生成任务出错: {str(e)}")
        # 更新任务状态为失败
        status_data = {
            "status": "failed",
            "progress": 0,
            "message": f"图像生成失败: {str(e)}"
        }
        await set_task_status(request_id, status_data)
        
        # 保存错误结果
        result_data = {
            "status": "failed",
            "message": f"图像生成失败: {str(e)}",
            "created_at": datetime.now().isoformat()
        }
        await set_task_result(request_id, result_data)


@router.post("/generate", response_model=ImageGenerationResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks
) -> Any:
    """
    根据文本描述生成图像
    
    流程：
    1. 用户提交描述文本
    2. LLM分析用户意图
    3. 设计细化方案
    4. 生成详细提示词
    5. 调用图像生成模型
    6. 返回生成的图像
    """
    try:
        # 详细记录请求信息
        logger.info("="*50)
        logger.info(f"【API端点】收到图像生成请求: 提示词长度={len(request.prompt)}，前50字符='{request.prompt[:50]}...'")
        logger.info(f"【API端点】请求参数: model={getattr(request, 'model', 'default')}，temperature={getattr(request, 'temperature', 0.7)}")
        logger.info("="*50)
        
        # 创建请求ID
        request_id = f"gen_{uuid.uuid4().hex[:12]}"
        logger.info(f"【API端点】已创建请求ID: {request_id}")
        
        # 启动后台任务
        background_tasks.add_task(
            process_image_generation,
            request_id=request_id,
            prompt=request.prompt
        )
        logger.info(f"【API端点】已添加后台任务: request_id={request_id}")
        
        # 返回初始响应
        return ImageGenerationResponse(
            status="processing",
            message="图像生成请求已提交，正在处理中",
            request_id=request_id,
            estimated_time=30  # 预计处理时间（秒）
        )
    
    except Exception as e:
        logger.error(f"【API端点】图像生成过程中发生错误: {str(e)}")
        # 打印详细堆栈信息
        import traceback
        logger.error(f"【API端点】错误堆栈: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像生成失败: {str(e)}"
        )


@router.post("/generate_with_image", response_model=ImageGenerationResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_with_image(
    prompt: str = Form(...),
    images: List[UploadFile] = File(None),
    model: str = Form("gemini-1.5-pro"),
    temperature: float = Form(0.7),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Any:
    """
    使用文本描述和上传的图像生成新图像并合成
    """
    try:
        # 详细记录请求信息
        logger.info("="*50)
        logger.info(f"【API端点】收到图像生成请求(带图像上传): 提示词长度={len(prompt)}，前50字符='{prompt[:50]}...'")
        logger.info(f"【API端点】请求参数: model={model}，temperature={temperature}，图像数量={len(images) if images else 0}")
        logger.info("="*50)
        
        # 创建请求ID
        request_id = f"gen_{uuid.uuid4().hex[:12]}"
        logger.info(f"【API端点】已创建请求ID: {request_id}")
        
        # 处理上传的图像
        image_paths = []
        if images:
            for i, img in enumerate(images):
                # 创建临时文件名
                file_ext = os.path.splitext(img.filename)[1]
                temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_id}_{i}{file_ext}")
                
                # 保存上传的文件
                with open(temp_file_path, "wb") as f:
                    content = await img.read()
                    f.write(content)
                    
                image_paths.append(temp_file_path)
                logger.info(f"【API端点】已保存上传图像 {i+1}/{len(images)}: {temp_file_path}")
                
        # 启动后台任务
        background_tasks.add_task(
            process_image_generation,
            request_id=request_id,
            prompt=prompt,
            image_paths=image_paths,
            model=model,
            temperature=temperature
        )
        logger.info(f"【API端点】已添加后台任务: request_id={request_id}, 图像数量={len(image_paths)}")
        
        # 返回初始响应
        return ImageGenerationResponse(
            status="processing",
            message="图像生成请求已提交，正在处理中",
            request_id=request_id,
            estimated_time=60 if images else 30  # 预计处理时间（秒）
        )
    
    except Exception as e:
        logger.error(f"【API端点】图像生成过程中发生错误: {str(e)}")
        # 打印详细堆栈信息
        import traceback
        logger.error(f"【API端点】错误堆栈: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像生成失败: {str(e)}"
        )


@router.get("/status/{request_id}", response_model=GenerationStatus)
async def get_generation_status(request_id: str) -> Any:
    """获取图像生成任务的状态"""
    status_data = await get_task_status(request_id)
    
    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"找不到请求ID: {request_id}"
        )
    
    return status_data


@router.get("/result/{request_id}", response_model=ImageGenerationResponse)
async def get_generation_result(request_id: str) -> Any:
    """获取图像生成的结果"""
    result_data = await get_task_result(request_id)
    
    if not result_data:
        # 检查是否任务仍在进行中
        status_data = await get_task_status(request_id)
        if status_data:
            return ImageGenerationResponse(
                status=status_data["status"],
                message=status_data["message"],
                request_id=request_id,
                estimated_time=30  # 预计还需要的处理时间（秒）
            )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"找不到请求ID: {request_id}"
        )
    
    return ImageGenerationResponse(
        status=result_data["status"],
        message=result_data["message"],
        request_id=request_id,
        images=result_data.get("images", []),
        created_at=datetime.fromisoformat(result_data["created_at"])
    ) 
