import re
from typing import List, Set, Tuple
from fastapi import Request, HTTPException
import json
from loguru import logger
from fastapi.responses import JSONResponse

class ContentFilterMiddleware:
    """
    内容安全过滤中间件，用于检测并阻止含有危险、违法犯罪词汇与意图的请求
    """
    
    def __init__(self, app):
        # 保存应用实例
        self.app = app
        
        # 初始化敏感词汇库
        self.sensitive_words: Set[str] = self._load_sensitive_words()
        
        # 敏感词汇分类
        self.violence_words = {
            "暴力", "杀人", "自杀", "杀害", "残害", "虐待", "伤害", "恐吓", "威胁", 
            "爆炸", "炸弹", "枪支", "武器", "屠杀", "血腥", "砍杀", "袭击", "轰炸"
        }
        
        self.illegal_words = {
            "毒品", "冰毒", "海洛因", "大麻", "摇头丸", "K粉", "违禁品", "走私", 
            "贩毒", "制毒", "吸毒", "贩卖", "制造", "犯罪", "违法", "偷窃", "盗窃",
            "抢劫", "诈骗", "作案", "洗钱", "黑客", "窃取", "加密货币"
        }
        
        self.adult_words = {
            "色情", "裸露", "性爱", "淫秽", "情色", "露骨", "色诱", "调情", "性虐待",
            "援交", "卖淫", "嫖娼", "性交易", "一夜情", "裸聊", "偷拍", "色情网站"
        }
        
        self.gambling_words = {
            "赌博", "博彩", "赌场", "赌钱", "彩票", "赌博网站", "老虎机", "赌注",
            "庄家", "赌局", "投注", "赛马", "赌球", "赌石", "赌资", "赢钱"
        }
        
        # 合并所有敏感词
        all_categories = [self.violence_words, self.illegal_words, self.adult_words, self.gambling_words]
        for category in all_categories:
            self.sensitive_words.update(category)
        
        # 危险行为意图正则表达式
        self.dangerous_patterns = [
            r'(如何|怎么|怎样).*(制造|制作|合成|购买|获取).*(炸弹|毒品|违禁品|武器)',
            r'(如何|怎么|怎样).*(偷窃|盗窃|抢劫|杀人|伤害|恐吓|侵犯|骗取|诈骗)',
            r'(色情|裸露|暴露|露出).*(儿童|未成年|小孩)',
            r'(自杀|轻生|结束生命).*(方法|办法|步骤|教程)',
            r'(贩卖|制作|购买|吸食).*(毒品|违禁药品|致幻剂)',
            r'(赌博|博彩).*(技巧|方法|窍门|平台)',
            r'(黑入|入侵|攻击).*(系统|网站|账号|设备)',
        ]
        
        logger.info(f"内容安全过滤中间件初始化完成，已加载 {len(self.sensitive_words)} 个敏感词")
    
    def _load_sensitive_words(self) -> Set[str]:
        """加载敏感词汇库"""
        try:
            # 尝试从文件加载
            with open("data/sensitive_words.txt", "r", encoding="utf-8") as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            # 如果文件不存在，使用默认列表
            return {
                "毒品", "炸弹", "色情", "赌博", "诈骗", "自杀", "暴力", "恐怖", 
                "违禁品", "武器", "黄赌毒", "非法", "犯罪", "邪教", "反动", 
                "裸露", "偷拍", "杀人", "杀害", "伤害", "伤亡", "袭击", "偷窃",
                "盗窃", "抢劫", "勒索", "绑架", "胁迫", "爆炸", "爆破", "枪支",
                "窃密", "黑客", "攻击", "入侵", "病毒", "木马", "劫持", "欺诈",
                "贿赂", "洗钱", "贩卖", "走私", "卖淫", "嫖娼", "性交易", "性虐待"
            }
    
    async def __call__(self, scope, receive, send):
        """
        ASGI接口调用方法 - 兼容版
        """
        # 只处理HTTP请求
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # 只过滤POST请求且是图像生成相关的路径
        method = scope.get("method", "")
        path = scope.get("path", "")
        
        if method != "POST" or not (path.endswith("/generate") or path.endswith("/generate_with_image")):
            # 不需要过滤的请求直接传递
            await self.app(scope, receive, send)
            return
        
        # 缓存请求体以便重复使用
        async def receive_and_cache():
            nonlocal cached_body, more_body
            
            if cached_body is not None:
                # 已缓存，返回缓存的消息
                return {
                    "type": "http.request",
                    "body": cached_body,
                    "more_body": more_body
                }
            
            # 接收原始消息
            message = await receive()
            
            # 如果是请求体，缓存它
            if message["type"] == "http.request":
                cached_body = message.get("body", b"")
                more_body = message.get("more_body", False)
                
                # 尝试提取和检查内容
                if cached_body:
                    try:
                        body_str = cached_body.decode('utf-8', errors='ignore')
                        logger.debug(f"接收到请求体大小: {len(body_str)} 字节")
                        
                        # 尝试提取提示词
                        prompt = ""
                        try:
                            # 尝试解析JSON
                            data = json.loads(body_str)
                            if "prompt" in data:
                                prompt = data["prompt"]
                        except json.JSONDecodeError:
                            # 不是JSON，尝试从表单中提取
                            if b"prompt=" in cached_body:
                                parts = cached_body.split(b"prompt=")
                                if len(parts) > 1:
                                    prompt_part = parts[1].split(b"&")[0]
                                    prompt = prompt_part.decode('utf-8', errors='ignore')
                        
                        # 检查内容安全
                        if prompt:
                            is_safe, reason = self._check_safety(prompt)
                            if not is_safe:
                                logger.warning(f"检测到不安全内容: '{prompt[:50]}...', 原因: {reason}")
                                # 标记请求为不安全
                                scope["_content_unsafe"] = True
                                scope["_unsafe_reason"] = reason
                    except Exception as e:
                        logger.error(f"内容过滤过程中出错: {str(e)}")
            
            return message
        
        # 劫持发送函数以拦截不安全内容的响应
        async def send_with_filter(message):
            if message["type"] == "http.response.start":
                # 检查是否标记为不安全
                if scope.get("_content_unsafe", False):
                    # 替换为403响应
                    reason = scope.get("_unsafe_reason", "未知原因")
                    logger.warning(f"拦截不安全内容，返回403状态码，原因: {reason}")
                    
                    # 创建新的403响应
                    new_message = {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [
                            (b"content-type", b"application/json"),
                        ]
                    }
                    await send(new_message)
                    
                    # 发送响应体
                    body = json.dumps({
                        "detail": f"请求包含不安全内容: {reason}"
                    }).encode()
                    
                    await send({
                        "type": "http.response.body",
                        "body": body,
                        "more_body": False
                    })
                    return
            
            # 正常传递其他消息
            await send(message)
        
        # 初始化缓存
        cached_body = None
        more_body = False
        
        # 使用自定义的receive和send
        await self.app(scope, receive_and_cache, send_with_filter)
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本，移除可能用于绕过过滤的特殊字符
        """
        # 移除常见的分隔符和干扰字符
        cleaned = re.sub(r'[\s\-_\.,:;!@#$%^&*()<>\[\]{}|~`+=\'\"?]+', '', text)
        return cleaned
    
    def _check_safety(self, text: str) -> Tuple[bool, str]:
        """
        检查文本是否安全，支持检测试图绕过过滤的情况
        
        Returns:
            tuple: (是否安全, 不安全原因)
        """
        if not text:
            return True, ""
        
        # 原始文本检查
        # 检查敏感词
        for word in self.sensitive_words:
            if word in text:
                return False, f"包含敏感词: {word}"
        
        # 清理后的文本检查（移除特殊字符）
        cleaned_text = self._clean_text(text)
        
        # 对清理后的文本再次检查敏感词
        for word in self.sensitive_words:
            if word in cleaned_text:
                return False, f"尝试规避过滤，实际包含敏感词: {word}"
        
        # 使用正则表达式检查隐藏的模式
        # 创建可以匹配中间带特殊字符的敏感词正则
        for word in self.sensitive_words:
            if len(word) > 1:  # 只检查长度大于1的词，避免误判
                # 创建一个正则表达式，允许词中间插入任何非字母数字的字符
                pattern = ''.join([c + r'[\s\W_]*' for c in word[:-1]]) + word[-1]
                if re.search(pattern, text, re.IGNORECASE):
                    return False, f"使用特殊字符分隔敏感词: {word}"
        
        # 检查危险模式
        for pattern in self.dangerous_patterns:
            # 在原始文本中检查
            if re.search(pattern, text):
                return False, f"匹配危险模式: {pattern}"
            
            # 在清理后的文本中检查
            if re.search(pattern, cleaned_text):
                return False, f"尝试规避过滤，实际匹配危险模式"
        
        return True, ""
