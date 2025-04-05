import time
import asyncio
from fastapi import Request, HTTPException
import redis.asyncio as redis
from loguru import logger
from app.core.config import settings
from fastapi.responses import JSONResponse

class RedisRequestLimiterMiddleware:
    """
    基于Redis的请求限制中间件，确保同一时间只有一个图像生成请求在处理
    简化版，专注于修复请求传递问题
    """
    
    def __init__(self, app, lock_timeout: int = 300):
        """初始化Redis请求限制中间件"""
        self.app = app
        self.lock_timeout = lock_timeout
        self.lock_key = "visionweaver:request_lock"
        
        # 从settings构建Redis URL
        auth_part = ""
        if getattr(settings, "REDIS_PASSWORD", None):
            auth_part = f":{settings.REDIS_PASSWORD}@"
        self.redis_url = f"redis://{auth_part}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
        
        # 初始化Redis客户端为None，会在第一次使用时初始化
        self.redis_client = None
        self.initialized = False
        
        logger.info("Redis请求限制中间件初始化（简化版）")
    
    async def init_redis(self):
        """初始化Redis连接"""
        if not self.initialized:
            try:
                self.redis_client = await redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
                self.initialized = True
                logger.info(f"已连接到Redis服务器: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            except Exception as e:
                logger.error(f"连接Redis失败: {str(e)}")
                self.initialized = False
    
    async def acquire_lock(self, request_id: str) -> bool:
        """尝试获取锁"""
        if not self.initialized:
            await self.init_redis()
            if not self.initialized:
                logger.warning("Redis未初始化，请求限制已禁用")
                return True
        
        lock_acquired = await self.redis_client.set(
            self.lock_key, 
            request_id, 
            nx=True,  # 只有在key不存在时才设置
            ex=self.lock_timeout  # 超时时间(秒)
        )
        
        if lock_acquired:
            logger.info(f"请求 {request_id} 获取锁成功")
        else:
            logger.info(f"请求 {request_id} 获取锁失败，有其他请求正在处理中")
        
        return lock_acquired
    
    async def release_lock(self, request_id: str) -> bool:
        """释放锁"""
        if not self.initialized:
            return False
        
        # 简化版：直接删除锁，不使用Lua脚本
        try:
            await self.redis_client.delete(self.lock_key)
            logger.info(f"请求 {request_id} 锁已释放")
            return True
        except Exception as e:
            logger.error(f"释放锁出错: {str(e)}")
            return False
    
    async def __call__(self, scope, receive, send):
        """
        ASGI接口调用方法 - 简化版
        """
        # 只处理HTTP请求
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # 只限制图像生成相关的路径
        path = scope.get("path", "")
        method = scope.get("method", "")
        
        # 检查是否需要限流
        should_limit = (method == "POST" and 
                        (path.endswith("/generate") or 
                         path.endswith("/generate_with_image")))
        
        if not should_limit:
            # 不需要限流的请求直接传递
            logger.debug(f"请求不需要限流: 方法={method}, 路径={path}")
            await self.app(scope, receive, send)
            return
        
        # 生成请求ID
        request_id = f"req_{int(time.time() * 1000)}"
        
        # 保存请求开始时间
        start_time = time.time()
        logger.info(f"处理图像生成请求: {request_id}, 路径={path}")
        
        # 尝试获取锁
        lock_acquired = await self.acquire_lock(request_id)
        
        if not lock_acquired:
            # 如果无法获取锁，返回429状态码
            logger.warning(f"请求 {request_id} 无法获取锁，返回429状态码")
            
            # 创建响应
            response = JSONResponse(
                status_code=429,
                content={"detail": "系统正在处理其他图像生成请求，请稍后再试"}
            )
            
            # 发送响应
            await response(scope, receive, send)
            return
        
        # 简化的请求跟踪
        async def wrapped_receive():
            message = await receive()
            message_type = message.get("type", "")
            
            # 记录请求体接收
            if message_type == "http.request":
                body_length = len(message.get("body", b""))
                logger.debug(f"请求 {request_id} 接收到请求体, 大小: {body_length} 字节")
                
                # 确保不丢失请求消息中的其他字段
                logger.debug(f"请求 {request_id} 消息类型: {message_type}, more_body: {message.get('more_body', False)}")
            
            return message
        
        # 简化的响应跟踪
        async def wrapped_send(message):
            message_type = message.get("type", "")
            
            # 记录响应状态
            if message_type == "http.response.start":
                status_code = message.get("status", 0)
                logger.debug(f"请求 {request_id} 响应状态码: {status_code}")
            
            # 发送消息
            await send(message)
        
        try:
            # 有锁，处理请求
            logger.info(f"请求 {request_id} 获取锁成功，处理请求...")
            
            # 记录处理开始
            logger.info(f"请求 {request_id} 已传递给应用处理")
            
            # 调用应用
            await self.app(scope, wrapped_receive, wrapped_send)
            
            # 记录处理完成
            elapsed = time.time() - start_time
            logger.info(f"请求 {request_id} 处理完成，耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            # 处理异常
            elapsed = time.time() - start_time
            logger.error(f"请求 {request_id} 处理过程中出错: {str(e)}")
            
            # 打印堆栈信息
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            
            # 尝试返回500错误
            try:
                response = JSONResponse(
                    status_code=500,
                    content={"detail": f"服务器内部错误: {str(e)}"}
                )
                await response(scope, receive, send)
            except Exception as send_error:
                logger.error(f"发送错误响应失败: {str(send_error)}")
        
        finally:
            # 释放锁
            lock_released = await self.release_lock(request_id)
            if lock_released:
                logger.info(f"请求 {request_id} 锁已释放，总处理时间: {time.time() - start_time:.2f}秒")
            else:
                logger.warning(f"请求 {request_id} 锁释放失败")
