#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VisionWeaver 工作流引擎

基于LangGraph的工作流引擎，用于处理图像生成和设计请求
使用Google Gemini模型实现确定性工作流程
"""

import os
import yaml
import asyncio
import uuid
import re  # 添加正则表达式模块
from typing import Dict, List, Any, Optional, Union, Callable, Iterator, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.tools.image_generator import generate_image
from app.tools.image_designer import image_designer
from app.tools.oss_uploader import upload_image_to_oss
from loguru import logger


# 定义工作流状态类型
class WorkflowState(TypedDict):
    """工作流状态定义"""
    # 用户输入和当前消息
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    # 当前执行阶段
    current_stage: str
    # 设计分析结果
    design_result: Optional[Dict]
    # 图像生成结果
    image_result: Optional[Dict]
    # 事件日志
    events: List[Dict]
    # 工作流开始时间
    start_time: float
    # 最终输出
    output: Optional[str]
    # 错误信息
    error: Optional[str]
    # 请求ID
    request_id: Optional[str]
    # 用户传入的图片路径列表（如logo、二维码等）
    input_images: Optional[List[str]] 
    # 图片合成结果
    composed_image_result: Optional[Dict]


class VisionWeaverEngine:
    """VisionWeaver LangGraph工作流引擎"""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        tools: Optional[List[BaseTool]] = None,
        system_prompt_path: Optional[str] = None,
        with_memory: bool = True,
        print_debug: bool = False
    ):
        """
        初始化VisionWeaver工作流引擎
        
        Args:
            model_name: 使用的模型名称，如不指定则使用配置中的默认值
            temperature: 模型温度参数
            tools: 可选的工具列表，如不指定则使用默认工具
            system_prompt_path: 系统提示词文件路径，如不指定则使用默认路径
            with_memory: 是否启用内存/对话历史功能
            print_debug: 是否打印调试信息
        """
        # 初始化LLM
        self.model_name = model_name or "gemini-1.5-pro"
        self.temperature = temperature
        self.print_debug = print_debug
        
        # 加载系统提示词
        self.system_prompt_path = system_prompt_path or os.path.join(
            os.path.dirname(__file__), "prompt", "system.yml"
        )
        self.system_prompt: Dict[str, str] = self._load_system_prompt()
        
        # 初始化LLM（在加载系统提示词之后）
        self.llm = self._init_llm()
        
        # 初始化工具
        self.tools = tools or self._init_default_tools()
        
        # 初始化工作流图
        self.workflow = self._create_workflow()
        
        # 初始化内存系统
        self.with_memory = with_memory
        if with_memory:
            self.memory = MemorySaver()
            # 重新创建带内存的工作流
            self.workflow = self._create_workflow(checkpointer=self.memory)
        
        logger.info(f"VisionWeaver工作流引擎初始化完成，使用模型: {self.model_name}")
        logger.info(f"已加载 {len(self.tools)} 个工具")
    
    def _init_llm(self) -> BaseChatModel:
        """初始化Google Gemini语言模型"""
        # 检查是否配置了Google API key
        if not settings.GOOGLE_API_KEY:
            raise ValueError("未配置Google API密钥，请在环境变量或.env文件中设置GOOGLE_API_KEY")
            
        logger.info(f"正在初始化Google Gemini模型：{self.model_name}")
        
        # 创建ChatGoogleGenerativeAI实例
        return ChatGoogleGenerativeAI(
            model=self.model_name,  # 如 "gemini-1.5-pro"
            temperature=self.temperature,
            google_api_key=settings.GOOGLE_API_KEY,
            convert_system_message_to_human=True,  # Gemini不支持SystemMessage，自动转换系统信息到Human消息
            max_output_tokens=4096  # 设置最大输出token数
        )
    
    def _load_system_prompt(self) -> Dict:
        """加载系统提示词"""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                system_data = yaml.safe_load(f)
                # 返回整个数据字典，而不仅是system_prompt字段
                return system_data
        except Exception as e:
            logger.error(f"加载系统提示词失败: {str(e)}")
            # 提供一个基本的后备提示词字典
            return {
                "system_prompt": "你是VisionWeaver，一个专业的AI图像生成与设计助手。",
                "assessment_prompt": "判断用户输入是否需要生成图像。如果是，返回true；否则返回false。",
                "design_prompt": "分析用户需求，提供专业设计方案。",
                "generation_prompt": "根据设计方案生成高质量图像。"
            }
    
    def _init_default_tools(self) -> List[BaseTool]:
        """初始化默认工具集"""
        try:
            # 导入图像合成工具
            from app.tools.image_composer import compose_image, add_image_watermark
            
            tools = [
                generate_image,
                image_designer,
                upload_image_to_oss,
                compose_image,
                add_image_watermark
            ]
            
            logger.info("成功加载图像合成工具")
            return tools
        except ImportError as e:
            logger.warning(f"加载图像合成工具失败: {str(e)}，将使用基本工具集")
            return [
                generate_image,
                image_designer,
                upload_image_to_oss
            ]
    
    def _add_event(self, state: WorkflowState, event_type: str, details: Dict) -> WorkflowState:
        """添加事件到工作流状态"""
        # 计算从开始到现在的时间差
        elapsed = round(asyncio.get_event_loop().time() - state["start_time"], 2)
        
        # 创建事件记录
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            **details
        }
        
        # 添加到事件列表
        events = state.get("events", [])
        events.append(event)
        
        # 打印事件信息用于调试
        if self.print_debug:
            print(f"[{elapsed}s] {event_type}: {details.get('message', '')}")
        
        logger.debug(f"事件: {event_type}, 详情: {details}")
        
        # 更新状态
        new_state = state.copy()
        new_state["events"] = events
        return new_state
    
    async def _initial_assessment(self, state: WorkflowState) -> WorkflowState:
        """初步评估用户输入的阶段"""
        logger.info("开始初步评估阶段")
        logger.debug(f"初始状态: current_stage={state.get('current_stage')}, messages={len(state.get('messages', []))}")
        
        # 添加事件
        state = self._add_event(state, "stage_start", {"message": "开始初步评估", "stage": "initial_assessment"})
        
        # 获取评估提示词 - 注意转义JSON中的大括号
        assessment_prompt_template = self.system_prompt.get("assessment_prompt", """
        你是VisionWeaver图像生成助手的意图分析器。
        你的任务是判断用户输入是否与图像生成或设计相关。
        只有明确的图像生成/设计需求才返回"需要生成图像"，否则返回"不需要生成图像"并直接回答用户。

        回复格式：
        ```json
        {
          "requires_image": true或false,
          "explanation": "解释判断理由",
          "response": "如果不需要生成图像，这里给出对用户的直接回复"
        }
        ```
        """)
        
        # 调用LLM进行初步评估
        try:
            # 直接使用消息列表，而不是ChatPromptTemplate
            messages = [
                SystemMessage(content=assessment_prompt_template),
                HumanMessage(content=state["messages"][-1].content)
            ]
            
            # 直接调用LLM
            logger.debug(f"调用LLM进行初步评估, 输入消息数: {len(messages)}")
            assessment_response = await self.llm.ainvoke(messages)
            logger.debug(f"初步评估响应: {assessment_response.content[:100]}...")
            
            # 尝试提取JSON响应
            import json
            import re
            
            # 尝试从回复中提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', assessment_response.content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'{.*}', assessment_response.content, re.DOTALL)
            
            if json_match:
                json_content = json_match.group(1) if '```' in json_match.group(0) else json_match.group(0)
                logger.debug(f"提取到的JSON内容: {json_content[:100]}...")
                assessment_result = json.loads(json_content)
                logger.debug(f"解析的JSON结果: {assessment_result}")
            else:
                # 无法解析JSON，默认为需要图像生成
                logger.warning("无法从LLM响应中解析JSON，默认为需要图像生成")
                assessment_result = {
                    "requires_image": True,
                    "explanation": "无法解析LLM响应，默认为需要图像生成",
                    "response": ""
                }
            
            # 更新状态
            new_state = state.copy()
            new_state["assessment_result"] = assessment_result
            
            # 添加事件
            new_state = self._add_event(new_state, "assessment_complete", {
                "message": f"初步评估完成，需要图像生成: {assessment_result['requires_image']}",
                "requires_image": assessment_result["requires_image"],
                "explanation": assessment_result["explanation"]
            })
            
            # 如果不需要图像生成，设置输出并结束
            if not assessment_result["requires_image"]:
                new_state["output"] = assessment_result["response"]
                new_state["current_stage"] = "complete"  # 使用明确的状态名称
                logger.debug(f"不需要生成图像，设置current_stage={new_state['current_stage']}")
                new_state = self._add_event(new_state, "workflow_end", {
                    "message": "工作流结束: 不需要图像生成，直接回复用户"
                })
            else:
                # 需要继续工作流
                new_state["current_stage"] = "design_analysis"
                logger.debug(f"需要生成图像，设置current_stage={new_state['current_stage']}")
                
            return new_state
            
        except Exception as e:
            logger.error(f"初步评估阶段出错: {str(e)}")
            error_state = state.copy()
            error_state["error"] = f"初步评估失败: {str(e)}"
            error_state["output"] = "抱歉，我在理解您的需求时遇到了问题。请尝试重新描述您想要的图像，或使用更简单的语言。"
            error_state["current_stage"] = "error"  # 使用明确的状态名称
            logger.debug(f"初步评估出错，设置current_stage={error_state['current_stage']}")
            error_state = self._add_event(error_state, "error", {
                "message": f"初步评估出错: {str(e)}",
                "error": str(e)
            })
            return error_state
    
    async def _design_analysis(self, state: WorkflowState) -> WorkflowState:
        """执行设计分析的阶段"""
        logger.info("开始设计分析阶段")
        
        # 添加事件
        state = self._add_event(state, "stage_start", {"message": "开始设计分析", "stage": "design_analysis"})
        
        try:
            # 获取用户输入
            user_demand = state["messages"][-1].content
            
            # 调用image_designer工具
            state = self._add_event(state, "tool_start", {
                "message": "开始调用image_designer工具",
                "tool": "image_designer",
                "input": user_demand[:100] + "..." if len(user_demand) > 100 else user_demand
            })
            
            # 修复：正确使用ainvoke方法，将参数打包成字典，键名为user_demand
            logger.debug(f"调用image_designer工具，输入: {user_demand[:50]}...")
            design_result = await image_designer.ainvoke({"user_demand": user_demand})
            
            # 检查设计结果
            if "错误" in design_result:
                logger.warning(f"设计分析返回错误: {design_result['错误']}")
                error_state = state.copy()
                error_state["error"] = f"设计分析失败: {design_result['错误']}"
                error_state["output"] = f"抱歉，在分析您的设计需求时遇到了问题: {design_result['错误']}。请尝试提供更详细的描述。"
                error_state["current_stage"] = "error"  # 确保使用字符串而不是END常量
                error_state = self._add_event(error_state, "error", {
                    "message": f"设计分析返回错误: {design_result['错误']}",
                    "error": design_result["错误"]
                })
                return error_state
            
            logger.info("设计分析完成，获取到设计方案")
            
            # 更新状态
            new_state = state.copy()
            new_state["design_result"] = design_result
            new_state["current_stage"] = "image_generation"  # 确保是字符串而不是常量
            
            # 添加事件
            new_state = self._add_event(new_state, "tool_end", {
                "message": "设计分析完成",
                "tool": "image_designer",
                "result_keys": list(design_result.keys())
            })
            
            # 设计分析已完成，可以进入图像生成阶段
            return new_state
            
        except Exception as e:
            logger.error(f"设计分析阶段出错: {str(e)}")
            error_state = state.copy()
            error_state["error"] = f"设计分析失败: {str(e)}"
            error_state["output"] = "抱歉，在分析您的设计需求时遇到了技术问题。请稍后再试。"
            error_state["current_stage"] = "error"  # 确保使用字符串而不是END常量
            error_state = self._add_event(error_state, "error", {
                "message": f"设计分析出错: {str(e)}",
                "error": str(e)
            })
            return error_state
    
    async def _image_generation(self, state: WorkflowState) -> WorkflowState:
        """执行图像生成的阶段"""
        logger.info("开始图像生成阶段")
        
        # 添加事件
        state = self._add_event(state, "stage_start", {"message": "开始图像生成", "stage": "image_generation"})
        
        try:
            # 检查是否有设计结果
            if "design_result" not in state or not state["design_result"]:
                logger.error("没有设计结果可用于图像生成")
                error_state = state.copy()
                error_state["error"] = "缺少设计结果，无法生成图像"
                error_state["output"] = "抱歉，在准备生成图像时遇到了问题。系统未能获取到设计方案。"
                error_state["current_stage"] = "error"  # 确保使用字符串而不是END常量
                error_state = self._add_event(error_state, "error", {
                    "message": "缺少设计结果，无法生成图像",
                    "error": "missing_design_result"
                })
                return error_state
            
            # 从设计结果中提取需要的信息
            design_result = state["design_result"]
            
            # 获取用户原始输入，用于调试
            user_input = state["messages"][-1].content if state["messages"] else "无输入"
            logger.debug(f"原始用户输入: {user_input}")
            logger.debug(f"设计结果详情: {design_result}")

            # 过滤用户输入中与图像合成相关的内容
            filtered_user_input = user_input
            # 检查是否有输入图像需要进行合成
            if state.get("input_images"):
                # 移除与二维码/logo添加相关的描述
                filtered_user_input = re.sub(r'(?i)(添加|放置|合成|插入).*?(二维码|logo|标志|图片|图像)', '', user_input)
                filtered_user_input = re.sub(r'(?i)(右下角|左下角|右上角|左上角).*?(二维码|logo|标志)', '', filtered_user_input)
                filtered_user_input = re.sub(r'(?i)(把|将).*?(二维码|logo|标志).*?(放|添加|合成|插入)', '', filtered_user_input)
                
                # 记录过滤结果
                logger.info(f"原始输入: {user_input}")
                logger.info(f"过滤后输入: {filtered_user_input}")
            
            # 检查设计结果与用户输入的相关性（通用方法，无硬编码）
            def extract_key_terms(text):
                """提取文本中的关键名词和动词"""
                import jieba.analyse
                # 使用jieba提取关键词，如果无法导入则跳过检查
                try:
                    # 尝试提取关键词，取权重最高的几个
                    keywords = jieba.analyse.extract_tags(text, topK=5)
                    return keywords
                except:
                    logger.debug("无法导入jieba或提取关键词")
                    return []

            try:
                # 如果设计结果包含设计方向信息
                if "设计方向" in design_result and isinstance(design_result["设计方向"], str):
                    design_text = design_result["设计方向"]
                    # 提取用户输入和设计结果中的关键词
                    user_keywords = extract_key_terms(filtered_user_input)
                    design_keywords = extract_key_terms(design_text)
                    
                    # 记录关键词匹配情况
                    if user_keywords and design_keywords:
                        common_keywords = set(user_keywords) & set(design_keywords)
                        logger.debug(f"用户输入关键词: {user_keywords}")
                        logger.debug(f"设计结果关键词: {design_keywords}")
                        logger.debug(f"共同关键词: {common_keywords}")
                        
                        # 如果没有共同关键词，记录警告但不阻止流程
                        if not common_keywords and len(user_keywords) >= 2:
                            logger.warning(f"设计结果可能与用户输入不匹配! 未发现共同关键词")
            except Exception as e:
                # 关键词提取失败不应影响主流程
                logger.debug(f"关键词匹配检查失败: {str(e)}")
            
            # 获取提示词生成模板
            prompt_builder_template = self.system_prompt.get("prompt_builder_template", """
            你是VisionWeaver的提示词合成专家。你的任务是将设计分析结果转换为高质量的图像生成提示词。
            
            提取设计方案中的关键视觉元素，创建一个详细的提示词。
            提示词应包含：
            1. 主题内容 - 图像应展示什么
            2. 风格 - 艺术风格、摄影风格或设计风格
            3. 构图 - 如何安排视觉元素
            4. 色彩方案 - 主色调和配色
            5. 光线和氛围 - 光照条件和整体感觉
            6. 技术细节 - 如超现实主义、渲染风格、景深等
            
            你的回复应该是原始的完整提示词，不需要解释或额外说明。提示词长度应在100-300字之间。
            注意：直接返回提示词，不要加任何其他文字。
            """)
            
            # 使用更通用的上下文增强方法
            enhanced_content = f"""
用户原始请求: {filtered_user_input}

设计分析结果:
{design_result}

请确保仅基于上述用户请求和设计分析结果生成图像提示词。
生成的提示词必须：
1. 准确反映用户需求的主题和目的
2. 包含设计分析中的关键视觉元素
3. 不包含任何历史对话或之前请求的元素
4. 不要包含任何二维码、logo或水印元素，这些将在后续步骤中单独添加
5. 如果涉及中国传统节日，请确保体现其文化内涵和传统元素

请直接返回完整提示词，不要添加解释或说明。
"""

            prompt_builder_messages = [
                SystemMessage(content=prompt_builder_template),
                HumanMessage(content=enhanced_content)
            ]
            
            # 调用LLM生成图像提示词
            logger.debug("正在生成图像提示词...")
            prompt_response = await self.llm.ainvoke(prompt_builder_messages)
            
            # 获取提示词
            image_prompt = prompt_response.content.strip()
            logger.info(f"生成的图像提示词: {image_prompt[:100]}...")
            
            # 添加事件 - 提示词生成
            state = self._add_event(state, "prompt_created", {
                "message": "已生成图像提示词",
                "prompt": image_prompt[:100] + "..." if len(image_prompt) > 100 else image_prompt
            })
            
            # 调用图像生成工具
            state = self._add_event(state, "tool_start", {
                "message": "开始调用generate_image工具",
                "tool": "generate_image",
                "prompt": image_prompt[:100] + "..." if len(image_prompt) > 100 else image_prompt
            })
            
            # 修复：正确使用ainvoke方法，将参数打包成字典作为input参数
            logger.debug(f"调用generate_image工具，提示词: {image_prompt[:50]}...")
            image_result = await generate_image.ainvoke({
                "prompt": image_prompt,
                "size": "1024x1024",  # 默认尺寸
                "return_oss_url": True  # 要求返回可访问的OSS URL
            })
            
            # 检查图像生成结果
            if "错误" in image_result:
                logger.warning(f"图像生成返回错误: {image_result['错误']}")
                error_state = state.copy()
                error_state["error"] = f"图像生成失败: {image_result['错误']}"
                error_state["output"] = f"抱歉，在生成图像时遇到了问题: {image_result['错误']}。请尝试使用不同的描述。"
                error_state["current_stage"] = "error"  # 确保使用字符串而不是END常量
                error_state = self._add_event(error_state, "error", {
                    "message": f"图像生成返回错误: {image_result['错误']}",
                    "error": image_result["错误"]
                })
                return error_state
                
            logger.info("图像生成完成")
            
            # 更新状态
            new_state = state.copy()
            new_state["image_result"] = image_result
            
            # 确保图片URL信息可用 - 如果没有远程URL但有本地路径，将本地路径作为URL
            if ("图片URL" not in image_result or not image_result["图片URL"]) and "本地路径" in image_result:
                logger.info(f"没有远程URL，使用本地文件路径: {image_result['本地路径']}")
                # 创建file://协议URL
                local_file_url = f"file://{image_result['本地路径']}"
                # 更新图像结果
                image_result["图片URL"] = local_file_url
                image_result["URL类型"] = "本地文件"
                # 更新状态中的图像结果
                new_state["image_result"] = image_result
            
            new_state["current_stage"] = "complete"  # 确保使用字符串而不是END常量
            
            # 添加事件
            new_state = self._add_event(new_state, "tool_end", {
                "message": "图像生成完成",
                "tool": "generate_image",
                "result_keys": list(image_result.keys())
            })
            
            # 记录图片访问信息，帮助调试
            if "图片URL" in image_result:
                logger.info(f"图片URL: {image_result['图片URL']}")
            if "本地路径" in image_result:
                logger.info(f"本地路径: {image_result['本地路径']}")
            
            # 获取响应模板
            response_template_content = self.system_prompt.get("response_template", """
            你是VisionWeaver，一个专业的AI图像生成与设计助手。
            根据设计分析和图像生成结果，创建一个友好且信息丰富的回复给用户。
            包括以下内容：
            1. 确认图像已生成
            2. 简要描述生成的图像特点
            3. 强调图像的优势和独特之处
            
            保持回复简洁友好，不要说太多废话。重点是告诉用户图像已成功生成，以及在哪里可以查看/下载图像。
            """)
            
            # 直接使用消息列表
            response_messages = [
                SystemMessage(content=response_template_content),
                HumanMessage(content=f"""
请用中文回复用户。

设计分析结果: {design_result}

图像生成结果: {image_result}

请注意:
1. 回复必须是中文
2. 一定要提及图片位置，告诉用户如何查看/访问图片
3. 如果有本地路径，明确告知文件存储在哪里

请生成友好、专业的中文回复给用户。
""")
            ]
            
            # 生成最终回复
            logger.debug("调用LLM生成最终回复...")
            final_response = await self.llm.ainvoke(response_messages)
            logger.debug(f"生成的回复: {final_response.content[:100]}...")
            
            # 确保回复是中文，如果检测到可能是英文，添加提示语
            if len(re.findall(r'[a-zA-Z]', final_response.content)) > len(re.findall(r'[\u4e00-\u9fa5]', final_response.content)):
                logger.warning("检测到回复可能是英文，添加中文说明")
                chinese_note = "您的图片已成功生成！请在系统中查看生成的图片。"
                if "图片URL" in image_result:
                    chinese_note += f"\n图片地址: {image_result['图片URL']}"
                if "本地路径" in image_result:
                    chinese_note += f"\n本地保存路径: {image_result['本地路径']}"
                final_response_content = chinese_note + "\n\n" + final_response.content
            else:
                final_response_content = final_response.content
            
            # 设置输出
            new_state["output"] = final_response_content
            
            # 检查是否有输入图像需要合成
            if new_state.get("input_images"):
                logger.info("检测到有输入图像需要合成，设置下一阶段为图像合成")
                new_state["current_stage"] = "image_composition"
                new_state = self._add_event(new_state, "stage_transition", {
                    "message": "图像生成完成，准备进入图像合成阶段",
                    "next_stage": "image_composition"
                })
            else:
                # 没有图像需要合成，直接结束工作流
                new_state["current_stage"] = "complete"
                # 添加事件 - 工作流结束
                new_state = self._add_event(new_state, "workflow_end", {
                    "message": "工作流成功完成: 图像已生成"
                })
            
            return new_state
            
        except Exception as e:
            logger.error(f"图像生成阶段出错: {str(e)}")
            error_state = state.copy()
            error_state["error"] = f"图像生成失败: {str(e)}"
            error_state["output"] = "抱歉，在生成图像时遇到了技术问题。请稍后再试。"
            error_state["current_stage"] = "error"  # 确保使用字符串而不是END常量
            error_state = self._add_event(error_state, "error", {
                "message": f"图像生成出错: {str(e)}",
                "error": str(e)
            })
            return error_state
    
    async def _image_composition(self, state: WorkflowState) -> WorkflowState:
        """图像合成阶段：将用户提供的图像(如logo)合成到生成的图像中"""
        try:
            # 记录事件
            state = self._add_event(state, "stage_start", {
                "stage": "image_composition",
                "message": "开始图像合成阶段"
            })
            
            logger.info("开始图像合成阶段")
            
            # 检查用户是否提供了需要合成的图像
            if not state.get("input_images") or len(state["input_images"]) == 0:
                logger.error("没有用户提供的图像可用于合成")
                error_state = state.copy()
                error_state["error"] = "缺少用户提供的图像，无法进行图像合成"
                error_state["output"] = "抱歉，您没有提供需要合成的图像（如logo或二维码）。请提供至少一张图像用于合成。"
                error_state["current_stage"] = "error"
                error_state = self._add_event(error_state, "error", {
                    "message": "缺少用户提供的图像，无法进行图像合成",
                    "error": "missing_input_images"
                })
                return error_state
                
            if not state.get("image_result"):
                logger.error("没有生成的图像可用于合成")
                error_state = state.copy()
                error_state["error"] = "缺少图像生成结果，无法进行图像合成"
                error_state["output"] = "抱歉，在准备合成图像时遇到了问题。系统未能找到生成的图像。"
                error_state["current_stage"] = "error"
                error_state = self._add_event(error_state, "error", {
                    "message": "缺少图像生成结果，无法进行图像合成",
                    "error": "missing_image_result"
                })
                return error_state
            
            # 从图像生成结果中获取基础图像路径
            base_image_path = None
            
            # 首先检查本地路径
            if "本地路径" in state["image_result"]:
                base_image_path = state["image_result"]["本地路径"]
                logger.info(f"使用本地图像路径进行合成: {base_image_path}")
            # 如果没有本地路径但有URL，尝试下载图像
            elif "图片URL" in state["image_result"]:
                image_url = state["image_result"]["图片URL"]
                logger.info(f"本地路径不可用，尝试从URL下载图像: {image_url}")
                
                try:
                    # 导入必要的库
                    import aiohttp
                    import os
                    from datetime import datetime
                    
                    # 创建临时目录
                    temp_dir = "temp_images"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # 生成临时文件名
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    temp_image_path = f"{temp_dir}/temp_image_{timestamp}.png"
                    
                    # 下载图像
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as response:
                            if response.status == 200:
                                with open(temp_image_path, "wb") as f:
                                    f.write(await response.read())
                                logger.info(f"图像成功下载到临时文件: {temp_image_path}")
                                base_image_path = temp_image_path
                            else:
                                raise Exception(f"下载图像失败，HTTP状态码: {response.status}")
                except Exception as e:
                    logger.error(f"下载图像时出错: {str(e)}")
                    error_state = state.copy()
                    error_state["error"] = f"无法从URL下载图像: {str(e)}"
                    error_state["output"] = "抱歉，在准备合成图像时遇到了问题。系统无法下载生成的图像。"
                    error_state["current_stage"] = "error"
                    error_state = self._add_event(error_state, "error", {
                        "message": f"无法从URL下载图像: {str(e)}",
                        "error": "download_image_failed"
                    })
                    return error_state
            else:
                logger.error("生成的图像没有本地路径信息也没有URL信息")
                error_state = state.copy()
                error_state["error"] = "无法获取生成图像的本地路径或URL"
                error_state["output"] = "抱歉，在合成图像时遇到了问题。系统无法访问生成的图像。"
                error_state["current_stage"] = "error"
                error_state = self._add_event(error_state, "error", {
                    "message": "无法获取生成图像的本地路径或URL",
                    "error": "missing_image_path_or_url"
                })
                return error_state
            
            # 准备合成分析提示词
            composition_prompt_template = self.system_prompt.get("composition_prompt", """
            你是一个图像合成顾问。现在有一张生成的图像和一组用户提供的图像(如logo或二维码)需要合成。
            分析用户需求，确定最佳的图像合成方案。考虑以下因素:
            
            1. 用户可能希望在哪里放置logo或二维码
            2. 图像应该占据多大比例
            3. 是否需要调整透明度
            4. 可能的位置选项有: top_left, top_right, bottom_left, bottom_right, center
            
            请分析用户输入，并提供图像合成的建议方案。
            """)
            
            # 获取所有输入图像信息
            input_images = state["input_images"]
            
            # 准备分析提示
            user_input = state["messages"][-1].content if state["messages"] else ""
            num_images = len(input_images)
            
            # 构建合成分析提示
            composition_analysis_prompt = f"""
            用户需求: {user_input}
            
            用户提供了{num_images}张图像用于合成:
            """
            
            # 添加每张输入图像的信息
            for i, img_path in enumerate(input_images):
                composition_analysis_prompt += f"\n图像{i+1}: {os.path.basename(img_path)}"
            
            composition_analysis_prompt += "\n\n请分析用户需求，确定如何将这些图像合成到生成的图像中。"
            
            # 记录工具调用事件
            state = self._add_event(state, "tool_start", {
                "tool": "composition_analyzer",
                "message": "开始分析图像合成需求",
                "input": composition_analysis_prompt
            })
            
            # 构建LLM消息
            messages = [
                SystemMessage(content=composition_prompt_template),
                HumanMessage(content=composition_analysis_prompt)
            ]
            
            try:
                # 调用LLM进行合成分析
                logger.info("调用LLM分析图像合成需求")
                composition_analysis = await self.llm.ainvoke(messages)
                
                # 获取合成分析结果
                composition_plan = composition_analysis.content
                
                # 记录工具调用结束事件
                state = self._add_event(state, "tool_end", {
                    "tool": "composition_analyzer",
                    "message": "完成图像合成需求分析",
                    "output": composition_plan[:100] + "..." if len(composition_plan) > 100 else composition_plan
                })
                
                # 解析合成方案
                # 这里我们将直接使用第一张用户图像进行合成
                # 未来可以根据LLM分析结果自动选择位置、大小等参数
                
                overlay_image_path = input_images[0]
                position = "bottom_right"  # 默认位置
                size = 0.2  # 默认大小
                
                # 尝试从分析结果中提取位置信息
                position_match = re.search(r'位置[：:]\s*(top_left|top_right|bottom_left|bottom_right|center)', composition_plan)
                if position_match:
                    position = position_match.group(1)
                    logger.info(f"从分析结果中提取到合成位置: {position}")
                
                # 尝试从分析结果中提取大小信息
                size_match = re.search(r'大小(?:比例)?[：:]\s*(\d+(?:\.\d+)?)\s*%', composition_plan)
                if size_match:
                    size_percent = float(size_match.group(1))
                    size = size_percent / 100.0
                    logger.info(f"从分析结果中提取到合成大小: {size} (原始: {size_percent}%)")
                
                # 记录工具调用事件
                state = self._add_event(state, "tool_start", {
                    "tool": "compose_image",
                    "message": "开始执行图像合成",
                    "input": {
                        "base_image": os.path.basename(base_image_path),
                        "overlay_image": os.path.basename(overlay_image_path),
                        "position": position,
                        "size": size
                    }
                })
                
                # 导入合成工具
                from app.tools.image_composer import compose_image
                
                # 执行图像合成
                logger.info(f"执行图像合成: 基础图像={base_image_path}, 叠加图像={overlay_image_path}, 位置={position}, 大小={size}")
                
                compose_result = await compose_image.ainvoke({
                    "base_image_path": base_image_path,
                    "overlay_image_path": overlay_image_path,
                    "position": position,
                    "overlay_size": size,
                    "return_oss_url": True  # 总是获取OSS URL
                })
                
                # 记录工具调用结束事件
                state = self._add_event(state, "tool_end", {
                    "tool": "compose_image",
                    "message": "完成图像合成",
                    "output": compose_result
                })
                
                # 检查合成结果
                if "错误" in compose_result:
                    logger.error(f"图像合成失败: {compose_result['错误']}")
                    error_state = state.copy()
                    error_state["error"] = f"图像合成失败: {compose_result['错误']}"
                    error_state["output"] = f"抱歉，在合成图像时遇到了问题: {compose_result['错误']}"
                    error_state["current_stage"] = "error"
                    error_state = self._add_event(error_state, "error", {
                        "message": f"图像合成失败: {compose_result['错误']}",
                        "error": "composition_failed"
                    })
                    return error_state
                
                # 合成成功，更新状态
                logger.info("图像合成成功")
                new_state = state.copy()
                new_state["composed_image_result"] = compose_result
                new_state["current_stage"] = "complete"
                
                # 构建完整回复
                # 准备最终回复的提示词
                final_response_prompt = self.system_prompt.get("final_response_prompt", """
                你是VisionWeaver，一个专业的图像生成和合成助手。
                
                请根据以下信息，为用户生成一个友好、专业的回复:
                
                1. 用户的原始需求
                2. 设计分析结果
                3. 图像生成结果
                4. 图像合成结果
                
                回复应该：
                1. 使用中文回复
                2. 简明扼要地概述生成和合成的过程
                3. 告知用户生成和合成的图像位置
                4. 提供一些关于图像内容和设计理念的描述
                5. 询问用户是否满意或需要进一步调整
                """)
                
                # 构建问题
                final_prompt = f"""
                用户原始需求: {user_input}
                
                设计分析结果: {state.get("design_result", {}).get("设计方案", "无设计分析")}
                
                图像生成信息:
                - 尺寸: {state.get("image_result", {}).get("图片尺寸", "未知")}
                - URL: {state.get("image_result", {}).get("图片URL", "无URL")}
                
                图像合成信息:
                - 使用的图像: {os.path.basename(overlay_image_path)}
                - 合成位置: {position}
                - 合成图像URL: {compose_result.get("图片URL", "无URL")}
                
                请生成一个友好的回复，告知用户图像已成功生成和合成，并提供相关细节。回复必须使用中文。
                """
                
                # 构建消息
                messages = [
                    SystemMessage(content=final_response_prompt),
                    HumanMessage(content=final_prompt)
                ]
                
                # 调用LLM生成最终回复
                final_response = await self.llm.ainvoke(messages)
                
                # 更新输出
                new_state["output"] = final_response.content
                
                # 记录工作流结束事件
                new_state = self._add_event(new_state, "workflow_end", {
                    "message": "工作流执行完成",
                    "status": "complete"
                })
                
                return new_state
                
            except Exception as e:
                logger.exception(f"图像合成分析过程中发生错误: {str(e)}")
                error_state = state.copy()
                error_state["error"] = f"图像合成分析过程中发生错误: {str(e)}"
                error_state["output"] = "抱歉，在分析图像合成需求时遇到了问题。请稍后重试或提供更具体的合成需求。"
                error_state["current_stage"] = "error"
                error_state = self._add_event(error_state, "error", {
                    "message": f"图像合成分析过程中发生错误: {str(e)}",
                    "error": "composition_analysis_failed"
                })
                return error_state
                
        except Exception as e:
            logger.exception(f"图像合成阶段发生错误: {str(e)}")
            error_state = state.copy()
            error_state["error"] = f"图像合成阶段发生错误: {str(e)}"
            error_state["output"] = "抱歉，图像合成过程中发生了问题。请稍后重试。"
            error_state["current_stage"] = "error"
            error_state = self._add_event(error_state, "error", {
                "message": f"图像合成阶段发生错误: {str(e)}",
                "error": "composition_stage_failed"
            })
            return error_state
    
    def _create_workflow(self, checkpointer=None):
        """创建工作流程图"""
        # 定义状态评估器（路由器）
        def router(state: WorkflowState) -> Union[str, Literal[END]]:
            """根据当前状态决定下一步"""
            # 明确检查字符串值，避免使用魔术字符串
            if state["current_stage"] == "error":
                logger.debug("路由器返回END - 因为当前阶段是'error'")
                return END  # 使用正确的END常量
            elif state["current_stage"] == "complete":
                logger.debug("路由器返回END - 因为当前阶段是'complete'")
                return END  # 使用正确的END常量
            else:
                logger.debug(f"路由器返回下一阶段: {state['current_stage']}")
                return state["current_stage"]
        
        # 创建工作流图
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("initial_assessment", self._initial_assessment)
        workflow.add_node("design_analysis", self._design_analysis)
        workflow.add_node("image_generation", self._image_generation)
        workflow.add_node("image_composition", self._image_composition)
        
        # 设置入口点
        workflow.set_entry_point("initial_assessment")
        
        # 添加边 - 基于路由函数
        workflow.add_conditional_edges(
            "initial_assessment",
            router,
            {
                "design_analysis": "design_analysis",
                END: END,  # 使用END常量作为键
                "error": END,
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "design_analysis",
            router,
            {
                "image_generation": "image_generation",
                END: END,  # 使用END常量作为键
                "error": END,
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "image_generation",
            router,
            {
                "image_composition": "image_composition",
                END: END,  # 使用END常量作为键
                "error": END,
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "image_composition",
            router,
            {
                END: END,  # 使用END常量作为键
                "error": END,
                "complete": END
            }
        )
        
        # 编译工作流
        if checkpointer:
            return workflow.compile(checkpointer=checkpointer)
        else:
            return workflow.compile()
    
    async def arun(
        self, 
        user_input: str, 
        thread_id: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        input_images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        异步执行工作流处理用户输入
        
        Args:
            user_input: 用户输入的文本
            thread_id: 对话线程ID，如启用了内存功能则需要提供
            callbacks: 可选的回调函数列表
            input_images: 可选的输入图像路径列表，用于图像合成（如logo、二维码等）
            
        Returns:
            包含工作流响应结果的字典
        """
        try:
            # 确保每次请求都有唯一的thread_id
            request_thread_id = thread_id or f"thread_{uuid.uuid4()}"
            logger.info(f"处理用户输入: {user_input[:50]}... (会话ID: {request_thread_id})")
            
            # 处理输入图像路径
            if input_images:
                # 验证所有图像路径是否存在
                valid_images = []
                for img_path in input_images:
                    if os.path.exists(img_path):
                        valid_images.append(img_path)
                        logger.info(f"添加用户提供的图像: {img_path}")
                    else:
                        logger.warning(f"图像路径不存在，忽略: {img_path}")
                        
                if valid_images:
                    logger.info(f"用户提供了 {len(valid_images)} 张有效图像用于合成")
                else:
                    logger.warning("所有提供的图像路径都无效，图像合成将被跳过")
                    input_images = None
            
            # 创建初始状态
            state = {
                "messages": [HumanMessage(content=user_input)],
                "current_stage": "initial_assessment",
                "design_result": None,
                "image_result": None,
                "events": [],
                "start_time": asyncio.get_event_loop().time(),
                "output": None,
                "error": None,
                "request_id": str(uuid.uuid4()),  # 添加唯一请求ID，确保状态隔离
                "input_images": valid_images if input_images else None,
                "composed_image_result": None
            }
            
            # 如果有输入图像，添加事件
            if state["input_images"]:
                state = self._add_event(state, "input_images_added", {
                    "message": f"用户提供了 {len(state['input_images'])} 张图像用于合成",
                    "image_count": len(state["input_images"]),
                    "image_paths": [os.path.basename(path) for path in state["input_images"]]
                })
            
            # 准备配置
            config = {
                # 禁用流式传输，使用直接调用模式
                "recursion_limit": 25,  # 设置最大递归限制
                "run_mode": "blocking",  # 使用阻塞模式而非流式
                "ensure_state_isolation": True  # 确保状态隔离
            }
            
            if self.with_memory and request_thread_id:
                if "configurable" not in config:
                    config["configurable"] = {}
                config["configurable"]["thread_id"] = request_thread_id
                logger.debug(f"使用会话ID: {request_thread_id}")
            
            if callbacks:
                config["callbacks"] = callbacks
            
            # 执行工作流
            logger.debug("开始执行工作流...")
            
            # 异步调用工作流 - 直接使用ainvoke模式，不使用astream
            result = await self.workflow.ainvoke(state, config=config)
            
            # 计算耗时
            elapsed = round(asyncio.get_event_loop().time() - state["start_time"], 2)
            logger.info(f"工作流执行完成，耗时: {elapsed}秒")
            
            # 构建返回结果
            response = {
                "output": result.get("output", "抱歉，处理过程中出现了问题。"),
                "events": result.get("events", []),
                "request_id": result.get("request_id", state["request_id"]),  # 保留请求ID
                "input_images": result.get("input_images", state["input_images"]),
                "composed_image_result": result.get("composed_image_result", state["composed_image_result"])
            }
            
            # 添加错误信息（如果有）
            if result.get("error"):
                response["error"] = result["error"]
                
            # 添加设计结果（如果有）
            if result.get("design_result"):
                response["design_result"] = result["design_result"]
                
            # 添加图像结果（如果有）
            if result.get("image_result"):
                response["image_result"] = result["image_result"]
            
            return response
        except Exception as e:
            logger.error(f"执行工作流时发生错误: {str(e)}")
            logger.exception("详细错误信息:")
            # 返回一个包含错误信息的结果，而不是抛出异常
            return {
                "output": f"抱歉，在处理您的请求时遇到了问题: {str(e)}。请再试一次或尝试其他描述方式。",
                "error": str(e)
            }
    
    def run(
        self, 
        user_input: str, 
        thread_id: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        input_images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        同步执行工作流处理用户输入
        
        Args:
            user_input: 用户输入的文本
            thread_id: 对话线程ID，如启用了内存功能则需要提供
            callbacks: 可选的回调函数列表
            input_images: 可选的输入图像路径列表，用于图像合成
            
        Returns:
            包含工作流响应结果的字典
        """
        # 创建事件循环
        loop = asyncio.new_event_loop()
        try:
            # 在事件循环中执行异步方法
            return loop.run_until_complete(
                self.arun(user_input, thread_id, callbacks, input_images)
            )
        finally:
            # 关闭事件循环
            loop.close()
        

# 创建全局引擎实例
vision_weaver_engine = VisionWeaverEngine(print_debug=True) 
