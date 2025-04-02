#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片设计机器人工具

基于用户文字描述来设计图片内容的专业机器人
使用langchain框架来实现，调用deepseek的api。基于用户的文生图需求描述，进行详尽的图片内容专业设计，并输出。
根据用户的需求意图，动态的调整温度参数，以确保生成图片符合用户需求。
"""

import json
import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any
from loguru import logger

from langchain_core.tools import tool
from langchain_core.tools import ToolException
# 修改导入，使用DeepSeek专用模型
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import settings


class ImageDesignerBot:
    """图片设计机器人，基于用户文字描述设计图片内容"""
    
    def __init__(self):
        """初始化图片设计机器人"""
        # 设置API密钥
        self.api_key = settings.DEEPSEEK_API_KEY or settings.OPENAI_API_KEY or settings.OPENROUTER_API_KEY
        
        if not self.api_key:
            logger.warning("未配置任何LLM API密钥，图片设计机器人不可用")
            self.is_configured = False
        else:
            self.is_configured = True
            
            # 设置系统提示语，定义机器人的角色和行为
            self.system_prompt = """你是一名专业的图像设计师，精通各种文化背景下的视觉艺术风格和设计原理。
你的任务是分析用户的图像需求描述，并提供专业、详细的图像设计方案。

当用户需求涉及特定文化、传统节日或习俗时，请特别注意：
1. 准确理解文化背景和内涵，尊重不同文化的传统表达
2. 识别传统节日的核心元素和象征符号
3. 选择文化上恰当的色彩、图案和象征物
4. 传达节日或传统活动的核心情感和氛围

分析时，你需要考虑以下方面：
1. 主题内容：用户想要表达的核心内容和主题
2. 视觉风格：适合的艺术风格、绘画技法或渲染风格
3. 构图原则：元素布局、视角、焦点位置等
4. 色彩方案：主色调、色彩搭配、色彩心理影响
5. 光影效果：光源位置、光线类型、阴影处理
6. 情感氛围：图像想要传达的情感和氛围
7. 技术要点：适合的渲染技术和特效
8. 文化元素：传统象征物、文化符号和习俗表达

请基于分析为用户提供两个不同风格的设计方案，确保方案具体、可视化且专业。每个方案都应该是独特的，并与用户需求匹配。
输出必须使用JSON格式，包含分析结果和两个设计方案。"""

            # 初始化基础提示模板
            self.prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(content="{input}")
            ])
            
            # 根据不同的API配置LLM
            self._setup_llm()
            
            logger.info("图片设计机器人初始化完成")
    
    def _setup_llm(self):
        """根据可用的API配置LLM"""
        if settings.DEEPSEEK_API_KEY:
            logger.info(f"使用DeepSeek模型: {settings.DEEPSEEK_MODEL}")
            self.llm = ChatDeepSeek(
                model=settings.DEEPSEEK_MODEL,
                api_key=settings.DEEPSEEK_API_KEY,
                temperature=0.7,
                streaming=False
            )
        else:
            logger.warning("未配置DeepSeek API密钥")
            raise ToolException("未配置DeepSeek API密钥，图片设计机器人不可用")
    
    async def _adjust_temperature(self, prompt: str) -> float:
        """
        根据用户需求自动调整温度参数
        
        高温度(0.8-1.0)：创意型、抽象型、艺术型需求
        中温度(0.5-0.7)：平衡型需求
        低温度(0.2-0.4)：精确型、写实型、技术型需求
        """
        # 创意关键词
        creative_keywords = ["创意", "想象", "梦幻", "奇幻", "抽象", "艺术", "独特", "新颖", "超现实"]
        # 精确关键词
        precise_keywords = ["精确", "写实", "照片级", "详细", "准确", "技术", "精细", "专业", "真实"]
        
        creative_score = sum(1 for word in creative_keywords if word in prompt)
        precise_score = sum(1 for word in precise_keywords if word in prompt)
        
        if creative_score > precise_score:
            temp = 0.8  # 高创意需求
        elif precise_score > creative_score:
            temp = 0.4  # 高精确需求
        else:
            temp = 0.7  # 平衡需求
            
        logger.info(f"根据用户需求调整温度参数为: {temp}")
        return temp
    
    async def analyze_prompt(self, prompt: str) -> Dict:
        """
        分析用户提示词，生成设计方案
        
        Args:
            prompt: 用户的图像描述文本
            
        Returns:
            包含分析结果和设计方案的字典
        """
        if not self.is_configured:
            raise ToolException("图片设计机器人未正确配置，请检查API密钥")
        
        try:
            # 动态调整温度参数
            temperature = await self._adjust_temperature(prompt)
            
            # 更新LLM的温度参数
            self.llm.temperature = temperature
            
            # 创建链
            chain = self.prompt_template | self.llm
            
            # 执行请求
            logger.info(f"开始分析用户提示词: {prompt[:50]}...")
            result = await chain.ainvoke({"input": prompt, "history": []})
            
            # 尝试解析JSON响应
            try:
                content = result.content
                # 提取JSON部分
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].strip()
                else:
                    json_content = content
                
                analysis_result = json.loads(json_content)
                logger.debug("设计方案生成完成")
                return analysis_result
            except Exception as e:
                logger.warning(f"JSON解析错误: {str(e)}")
                logger.warning(f"原始内容: {result.content[:200]}...")
                
                # 尝试格式化为固定结构
                return {
                    "分析结果": {
                        "主题内容": "无法解析",
                        "视觉风格": "无法解析",
                        "构图原则": "无法解析",
                        "色彩方案": "无法解析",
                        "光影效果": "无法解析"
                    },
                    "设计方案一": {
                        "标题": "格式化失败",
                        "描述": result.content[:500] + "..."
                    },
                    "设计方案二": {
                        "标题": "格式化失败",
                        "描述": "请重试或提供更详细的描述"
                    }
                }
        
        except Exception as e:
            logger.error(f"生成设计方案时发生错误: {str(e)}")
            raise ToolException(f"生成设计方案失败: {str(e)}")


# 创建全局实例
image_designer_bot = ImageDesignerBot() 


@tool
async def image_designer(user_demand: str) -> Dict:
    """基于用户文字描述来设计图片内容的专业机器人
    
    分析用户需求，提供专业的图像设计方案，包括主题内容、视觉风格、构图原则、色彩方案等关键元素。
    
    Args:
        user_demand: 用户对图片的详细描述和需求，包括用途、目标受众和风格偏好
        
    Returns:
        包含分析结果和多个设计方案的详细信息，可用于生成高质量图像
    """
    if not user_demand or len(user_demand.strip()) < 5:
        return {
            "错误": "请提供更详细的图片描述，至少5个字符"
        }
        
    try:
        logger.info(f"图像设计顾问开始分析用户需求: {user_demand[:50]}...")
        design_result = await image_designer_bot.analyze_prompt(user_demand)
        logger.info("图像设计分析完成，返回设计方案")
        return design_result
    except ToolException as e:
        # 工具异常直接传递
        logger.error(f"图像设计工具异常: {str(e)}")
        raise e
    except Exception as e:
        # 其他异常转换为工具异常
        logger.error(f"图片设计过程中发生未知错误: {str(e)}")
        raise ToolException(f"图片设计失败: {str(e)}") 