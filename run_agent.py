#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VisionWeaver 工作流测试入口

提供命令行界面与图像生成工作流交互
支持单次查询和交互式模式 - 使用Google Gemini模型
"""

import os
import sys
import uuid
import argparse
import asyncio
import time
import re
from typing import Optional, Dict, Any
import readline  # 用于提供输入历史
from datetime import datetime
from functools import wraps

from app.core.engine import VisionWeaverEngine
from app.core.config import settings
from loguru import logger


# 添加API速率限制处理装饰器
def rate_limiter(max_per_minute=1):
    """限制函数调用频率的装饰器"""
    last_called = [0.0]  # 使用列表以便在闭包中可修改
    min_interval = 60.0 / max_per_minute
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            elapsed = now - last_called[0]
            
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                print_colored(f"API速率限制: 等待 {wait_time:.1f} 秒...", "yellow")
                await asyncio.sleep(wait_time)
            
            try:
                result = await func(*args, **kwargs)
                last_called[0] = time.time()
                return result
            except Exception as e:
                error_str = str(e)
                # 检测速率限制错误
                if "429" in error_str and "quota" in error_str:
                    # 尝试从错误消息中提取重试延迟时间
                    retry_delay = 10  # 默认10秒
                    delay_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    if delay_match:
                        retry_delay = int(delay_match.group(1))
                        
                    print_colored(f"达到API速率限制，将在 {retry_delay} 秒后重试...", "yellow")
                    print_colored("Google Gemini API对免费用户有以下限制:", "cyan")
                    print_colored("- 每分钟请求数(RPM): 2", "cyan")
                    print_colored("- 每日请求数(RPD): 50", "cyan")
                    print_colored("- 每分钟令牌数(TPM): 32,000", "cyan")
                    print_colored("了解更多: https://ai.google.dev/gemini-api/docs/rate-limits", "cyan")
                    
                    await asyncio.sleep(retry_delay)
                    # 重试请求
                    print_colored("正在重试请求...", "green")
                    try:
                        return await func(*args, **kwargs)
                    except Exception as retry_error:
                        # 如果重试也失败，抛出原始错误
                        print_colored("重试失败，请稍后再试", "red")
                        raise retry_error
                else:
                    # 非速率限制错误，直接抛出
                    raise
        return wrapper
    return decorator


def setup_logging(debug: bool = False):
    """配置日志"""
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台日志
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level, 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    # 添加文件日志
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger.add(f"{log_dir}/visionweaver_{datetime.now().strftime('%Y%m%d')}.log", 
               rotation="10 MB", level="DEBUG")


def print_colored(text: str, color: str = "white"):
    """打印彩色文本"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "bold": "\033[1m",
        "end": "\033[0m"
    }
    
    print(f"{colors.get(color, '')}{text}{colors['end']}")


def print_welcome():
    """打印欢迎信息"""
    print_colored("\n" + "="*60, "cyan")
    print_colored("欢迎使用 VisionWeaver - AI图像生成与设计助手", "bold")
    print_colored(f"Powered by Google Gemini - 当前模型: {settings.AGENT_MODEL}", "cyan")
    print_colored("输入文字描述来生成图片，输入 'exit' 或 'quit' 退出", "cyan")
    print_colored("="*60 + "\n", "cyan")


async def handle_result(result: Dict[str, Any], show_events: bool = True):
    """处理并展示工作流结果"""
    # 检查结果是否有效
    if not result:
        print_colored("\n错误: 未收到任何结果", "red")
        print_colored("请检查日志获取更多信息并尝试再次运行", "yellow")
        return
    
    # 提取最终回复
    final_output = result.get("output")
    
    # 显示事件日志
    if show_events and "events" in result and result["events"]:
        print_colored("\n[工作流事件]:", "blue")
        for i, event in enumerate(result["events"]):
            event_type = event.get("type", "未知事件")
            
            # 根据事件类型选择颜色
            color = "white"
            if event_type == "stage_start":
                color = "cyan"
            elif event_type == "tool_start":
                color = "yellow"
            elif event_type == "tool_end":
                color = "green"
            elif event_type == "error":
                color = "red"
            elif event_type == "workflow_end":
                color = "magenta"
            
            # 格式化输出事件
            elapsed = event.get("elapsed_seconds", 0)
            message = event.get("message", "")
            print_colored(f"[{elapsed:.2f}s] {event_type}: {message}", color)
            
            # 对于某些事件类型，显示额外信息
            if event_type == "tool_start" and "tool" in event:
                print_colored(f"  工具: {event.get('tool')}", "white")
                if "input" in event:
                    print_colored(f"  输入: {event.get('input')}", "white")
    
    # 检查是否有图像生成结果
    if "image_result" in result:
        print_colored("\n[图像生成结果]:", "green")
        image_result = result["image_result"]
        
        # 显示图像URL
        if "图片URL" in image_result:
            print_colored(f"图片URL: {image_result['图片URL']}", "cyan")
        
        # 显示本地路径
        if "本地路径" in image_result:
            print_colored(f"本地路径: {image_result['本地路径']}", "cyan")
            
        # 显示图片尺寸和其他元数据
        if "图片尺寸" in image_result:
            print_colored(f"图片尺寸: {image_result['图片尺寸']}", "white")
    
    # 检查是否有图像合成结果
    if "composed_image_result" in result and result["composed_image_result"]:
        print_colored("\n[图像合成结果]:", "green")
        composed_result = result["composed_image_result"]
        
        # 显示合成图像URL
        if "图片URL" in composed_result:
            print_colored(f"合成图片URL: {composed_result['图片URL']}", "magenta")
        
        # 显示本地路径
        if "本地路径" in composed_result:
            print_colored(f"合成图片本地路径: {composed_result['本地路径']}", "magenta")
        
        # 显示合成信息
        if "合成位置" in composed_result:
            print_colored(f"合成位置: {composed_result['合成位置']}", "white")
        if "合成尺寸比例" in composed_result:
            print_colored(f"合成尺寸: {composed_result['合成尺寸比例']}", "white")
    
    # 检查是否有错误信息
    if "error" in result:
        print_colored("\n[错误信息]:", "red")
        print_colored(result["error"], "yellow")
    
    # 检查是否有输出
    if not final_output:
        print_colored("\n错误: 未收到AI回复内容", "red")
        print_colored("这可能是模型或工具调用问题，请查看日志了解详情", "yellow")
        return
        
    print_colored("\n" + "-"*60, "blue")
    print_colored("AI回复:", "bold")
    print(final_output)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="VisionWeaver - AI图像生成与设计助手")
    
    # 主要操作模式
    parser.add_argument("-q", "--query", type=str, help="单次查询模式，提供需求描述")
    parser.add_argument("-f", "--file", type=str, help="从文件读取需求描述 (支持.txt或.md文件)")
    
    # 模型设置
    parser.add_argument("--model", type=str, 
                       help=f"模型名称，默认为gemini-1.5-pro", 
                       default="gemini-1.5-pro")
    parser.add_argument("--temperature", type=float, default=0.7, help="模型温度参数，控制创造性 (0.1-1.0)")
    
    # 会话管理
    parser.add_argument("--thread-id", type=str, help="使用特定的对话线程ID继续之前的对话")
    parser.add_argument("--no-memory", action="store_true", help="禁用对话记忆功能")
    
    # 速率限制设置
    parser.add_argument("--rpm", type=float, default=1.0, help="每分钟最大请求数，用于处理API速率限制 (默认: 1)")
    
    # 输出控制
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息，包括工作流事件")
    parser.add_argument("--debug", action="store_true", help="启用调试模式，显示更多日志信息")
    
    # 图像合成功能
    parser.add_argument("--image", action="append", help="要合成到生成图像中的图片路径(如logo或二维码)，可指定多次添加多张图片")
    
    # 简化测试功能
    parser.add_argument("--test", action="store_true", help="测试模式，使用预定义的测试用例")
    
    return parser.parse_args()


def read_requirement_file(file_path):
    """从文件读取需求描述"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 记录文件内容长度
        logger.info(f"从文件 {file_path} 读取了 {len(content)} 字符的需求")
        
        # 检查文件是否为markdown，如果是，可以考虑提取正文内容
        if file_path.lower().endswith('.md'):
            # 简单处理一下，移除markdown标题和代码块标记
            # 这里只是基础处理，可以根据需要使用更复杂的markdown解析
            import re
            # 移除 # 开头的标题行
            content = re.sub(r'^#.*$', '', content, flags=re.MULTILINE)
            # 移除代码块标记
            content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            # 移除多余的空行
            content = re.sub(r'\n\s*\n', '\n\n', content)
            
            logger.info("已处理Markdown格式内容")
        
        return content
    except Exception as e:
        logger.error(f"读取需求文件时出错: {str(e)}")
        raise ValueError(f"无法读取需求文件 {file_path}: {str(e)}")


async def run_interactive_mode(engine: VisionWeaverEngine, args):
    """运行交互式模式"""
    print_welcome()
    
    # 不再在函数顶部创建会话ID，而是为每次请求创建新的
    print_colored("已进入交互式模式，每次请求将使用独立会话", "blue")
    print_colored("特殊命令:", "cyan")
    print_colored("  'exit'/'quit' - 退出程序", "cyan")
    print_colored("  'clear' - 清除历史", "cyan")
    print_colored("  'file <路径>' - 从文件读取需求", "cyan")
    
    # 图像合成功能说明
    if hasattr(args, 'image') and args.image:
        print_colored(f"\n已指定 {len(args.image)} 张图像用于合成:", "cyan")
        for img in args.image:
            print_colored(f"  - {img}", "cyan")
        print_colored("这些图像将用于所有对话中的图像合成", "cyan")
    
    try:
        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入您的需求 > ").strip()
                
                # 退出检查
                if user_input.lower() in ["exit", "quit", "退出", "q"]:
                    print_colored("感谢使用！再见！", "green")
                    break
                
                # 清除历史检查
                if user_input.lower() in ["clear", "清除", "c"]:
                    print_colored("已清除会话历史", "yellow")
                    continue
                
                # 从文件读取需求
                if user_input.lower().startswith("file ") or user_input.lower().startswith("文件 "):
                    file_path = user_input.split(" ", 1)[1].strip()
                    try:
                        user_input = read_requirement_file(file_path)
                        print_colored(f"已从文件 {file_path} 读取需求:", "green")
                        print_colored("-"*60, "cyan")
                        print(user_input)
                        print_colored("-"*60, "cyan")
                    except Exception as e:
                        print_colored(f"读取文件时出错: {str(e)}", "red")
                        continue
                
                # 检测多行输入并给予警告
                if "\n" in user_input:
                    print_colored("警告: 检测到多行输入!", "red")
                    print_colored("直接输入的多行文本可能导致执行错误。请使用文件输入来处理多行需求。", "yellow")
                    print_colored("您可以使用 'file <文件路径>' 命令从文件读取需求。", "cyan")
                    print_colored("是否仍要继续处理此多行输入? (y/n)", "yellow")
                    confirm = input("> ").strip().lower()
                    if confirm != 'y':
                        continue
                    
                # 跳过空白输入
                if not user_input:
                    continue
                
                # 为每次请求创建新的thread_id
                thread_id = str(uuid.uuid4())
                logger.debug(f"为当前请求创建新会话ID: {thread_id}")
                
                # 正常输出模式
                print_colored("AI正在思考...", "yellow")
                
                # 获取输入图像路径
                input_images = args.image if hasattr(args, 'image') and args.image else None
                
                result = await engine.arun(user_input, thread_id, input_images=input_images)
                await handle_result(result, args.verbose)
                    
            except KeyboardInterrupt:
                print_colored("\n\n操作被用户中断。输入'exit'退出或继续输入。", "yellow")
            except Exception as e:
                logger.exception("处理用户输入时发生错误")
                print_colored(f"\n发生错误: {str(e)}", "red")
                print_colored("请尝试不同的输入或重启程序。", "yellow")
    
    except Exception as e:
        logger.exception("交互式模式发生错误")
        print_colored(f"程序发生错误: {str(e)}", "red")
    
    print_colored("\n会话结束。", "blue")


async def run_single_query(engine: VisionWeaverEngine, query: str, args):
    """运行单次查询模式"""
    thread_id = args.thread_id or str(uuid.uuid4())
    
    try:
        # 检测多行输入并给予警告
        if "\n" in query:
            line_count = query.count('\n') + 1
            print_colored(f"检测到多行输入 ({line_count} 行)", "yellow")
            if not args.file:  # 如果不是从文件读取，则显示警告
                print_colored("注意: 直接在命令行中输入的多行文本可能导致执行错误。", "yellow")
                print_colored("建议使用 -f/--file 参数从文件读取需求。", "cyan")
                print_colored("是否继续处理此多行输入? (y/n)", "yellow")
                confirm = input("> ").strip().lower()
                if confirm != 'y':
                    print_colored("已取消操作", "red")
                    return
        
        # 正常输出模式
        print_colored("AI正在处理您的请求...", "yellow")
        
        # 检查是否有输入图像
        input_images = args.image if hasattr(args, 'image') and args.image else None
        if input_images:
            print_colored(f"将使用 {len(input_images)} 张图像进行合成", "cyan")
            for img in input_images:
                print_colored(f"  - {img}", "cyan")
        
        result = await engine.arun(query, thread_id, input_images=input_images)
        await handle_result(result, args.verbose)
        
        if not args.thread_id:
            print_colored(f"\n如需继续此对话，请使用会话ID: {thread_id}", "blue")
            
    except Exception as e:
        logger.exception("处理单次查询时发生错误")
        print_colored(f"发生错误: {str(e)}", "red")


async def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    setup_logging(args.debug)
    
    # 检查API密钥是否配置
    if not settings.GOOGLE_API_KEY:
        print_colored("错误: 未配置Google API密钥", "red")
        print_colored("请在环境变量或.env文件中设置GOOGLE_API_KEY", "yellow")
        return
    
    # 测试模式
    if args.test:
        print_colored("\n运行测试模式...", "cyan")
        test_queries = [
            "生成一只可爱的卡通猫咪图片",
            "什么是量子计算机？",  # 不需要图像的查询
            "帮我设计一个现代风格的公司logo，蓝色调为主"
        ]
        
        # 如果指定了图像，添加图像合成测试
        if hasattr(args, 'image') and args.image:
            test_queries.append("生成一个产品宣传图，并在右下角加上我们公司的logo")
            test_queries.append("设计一张海报，右下角需要放二维码方便扫码")
        
        # 初始化测试引擎
        engine = VisionWeaverEngine(
            model_name=args.model,
            temperature=args.temperature,
            with_memory=not args.no_memory,
            print_debug=True
        )
        
        # 运行测试用例
        for i, query in enumerate(test_queries):
            print_colored(f"\n测试 {i+1}/{len(test_queries)}: {query}", "cyan")
            await run_single_query(engine, query, args)
            # 暂停1秒，避免API限制
            await asyncio.sleep(1)
            
        print_colored("\n测试完成！", "green")
        return
    
    # 导入必要的库
    try:
        import langchain_google_genai
    except ImportError:
        print_colored("错误: 缺少必要的库 langchain_google_genai", "red")
        print_colored("正在尝试安装必要的库...", "yellow")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain_google_genai"])
            print_colored("安装成功！", "green")
            # 重新导入
            import langchain_google_genai
        except Exception as e:
            print_colored(f"安装失败: {str(e)}", "red")
            print_colored("请手动安装: pip install langchain_google_genai", "yellow")
            return
    
    # 初始化引擎
    try:
        engine = VisionWeaverEngine(
            model_name=args.model,
            temperature=args.temperature,
            with_memory=not args.no_memory,
            print_debug=args.debug  # 使用debug参数控制工作流事件打印
        )
        
        # 显示模型信息
        print_colored(f"使用Google Gemini模型: {args.model}", "blue")
        print_colored(f"温度参数: {args.temperature}", "blue")
        print_colored("已应用API速率限制管理，自动处理请求频率限制", "cyan")
        
        # 应用速率限制装饰器
        # 根据用户指定的请求间隔控制速率
        rpm_limit = args.rpm if hasattr(args, 'rpm') and args.rpm else 1
        logger.info(f"设置API请求速率限制为每分钟 {rpm_limit} 次")
        original_arun = engine.arun
        engine.arun = rate_limiter(max_per_minute=rpm_limit)(engine.arun)
        
        # 处理从文件读取需求的情况
        if args.file:
            try:
                query = read_requirement_file(args.file)
                print_colored(f"从文件 {args.file} 读取需求成功", "green")
                args.query = query  # 将文件内容作为查询内容
            except Exception as e:
                print_colored(f"读取需求文件时出错: {str(e)}", "red")
                return
        
        if args.query:
            # 单次查询模式
            await run_single_query(engine, args.query, args)
        else:
            # 交互式模式
            await run_interactive_mode(engine, args)
            
    except ValueError as e:
        print_colored(f"错误: {str(e)}", "red")
    except ImportError as e:
        print_colored(f"导入错误: {str(e)}", "red")
        print_colored("请确保所有必要的库已安装", "yellow")
    except Exception as e:
        logger.exception("初始化引擎时发生错误")
        print_colored(f"初始化失败: {str(e)}", "red")


if __name__ == "__main__":
    # 设置颜色支持
    os.system('')  # 启用VT100转义序列
    
    # 运行主函数
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_colored("\n程序被用户中断。再见！", "yellow")
    except Exception as e:
        logger.exception("程序运行异常")
        print_colored(f"\n程序发生异常: {str(e)}", "red") 