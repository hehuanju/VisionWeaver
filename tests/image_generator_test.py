#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片生成机器人测试脚本

用于测试image_generator.py中的图片生成功能
支持控制台输入的交互式测试
"""

import os
import sys
import json
import asyncio
import argparse
import re
from pathlib import Path
from loguru import logger

# 确保能够导入app模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入被测试的组件
from app.tools.image_generator import ImageGeneratorBot, image_generator_bot, generate_image
from app.core.config import settings

# 配置日志
logger.remove()  # 移除默认处理器
logger.add(sys.stderr, level="DEBUG")  # 添加stderr处理器，修改为DEBUG级别
logger.add("tests/image_generator_test.log", rotation="10 MB", level="DEBUG")  # 添加文件处理器


async def test_with_input(prompt: str, size: str = "1024x1024", return_oss_url: bool = False):
    """
    使用用户输入的描述测试图片生成
    
    Args:
        prompt: 用户输入的图片描述
        size: 生成图片的尺寸，格式为"宽x高"
        return_oss_url: 是否直接返回OSS URL
    """
    # 确保输入不为空
    if not prompt or len(prompt.strip()) == 0:
        print("\n错误: 请提供有效的图片描述，不能为空白")
        return
        
    logger.info(f"使用输入描述测试图片生成: {prompt[:50]}...")
    logger.info(f"图片尺寸: {size}, 直接返回OSS URL: {return_oss_url}")
    
    try:
        # 检查API密钥
        if not settings.GOOGLE_API_KEY:
            print("\n错误: 未配置Google API密钥，请在.env文件中设置GOOGLE_API_KEY")
            return
        
        # 显示进度
        print("\n正在使用Gemini模型生成图片，请稍候...\n")
        
        try:
            # 调用图片生成工具 - 使用ainvoke方法
            result = await generate_image.ainvoke({
                "prompt": prompt,
                "size": size,
                "return_oss_url": return_oss_url
            })
        except Exception as e:
            print(f"\n❌ 调用图片生成工具时发生错误: {str(e)}")
            logger.error(f"调用图片生成工具时发生错误: {str(e)}")
            return
        
        # 输出结果
        print("\n" + "="*50)
        print("图片生成结果")
        print("="*50)
        
        # 检查错误结果
        if not result:
            print("\n❌ 错误: 生成结果为空")
            return
            
        if isinstance(result, dict) and "错误" in result:
            print(f"\n❌ 错误: {result['错误']}")
            return
        
        # 输出信息
        if isinstance(result, dict):
            # 如果是直接返回OSS URL的简单结果
            if return_oss_url and "图片URL" in result:
                print(f"\n🌅 OSS图片URL: {result['图片URL']}")
                print("  (已直接返回OSS URL)")
                return
                
            # 处理标准结果
            if "使用模型" in result:
                print(f"\n📋 使用模型: {result['使用模型']}")
                
            if "生成来源" in result:
                print(f"\n📋 生成方式: {result['生成来源']}")
            
            if "图片尺寸" in result:
                print(f"\n📏 图片尺寸: {result['图片尺寸']}")
            
            # 处理图片URL
            if "图片URL" in result:
                print(f"\n🌅 图片URL: {result['图片URL']}")
                
                # 处理存储位置
                if "存储位置" in result:
                    print(f"  (存储位置: {result['存储位置']})")
                
                # 处理OSS路径
                if "OSS路径" in result:
                    print(f"  OSS路径: {result['OSS路径']}")
            
            # 处理图片数据
            if "图片数据" in result:
                print(f"\n📊 {result['图片数据']}")
            
            # 处理本地路径
            if "本地路径" in result:
                print(f"\n💾 本地保存路径: {result['本地路径']}")
                print("  (图片已保存到本地)")
            
            # 保存详细结果到文件
            os.makedirs("tests/results", exist_ok=True)
            timestamp = asyncio.get_event_loop().time()
            result_file = f"tests/results/image_result_{int(timestamp)}.json"
            
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n📄 结果信息已保存至: {result_file}")
            
            # 检查debug文件
            debug_files = [f for f in os.listdir() if f.startswith("debug_response_")]
            if debug_files:
                latest_debug_file = max(debug_files, key=os.path.getctime)
                print(f"\n🔍 调试信息文件: {latest_debug_file}")
                print("  可查看此文件了解API响应详情")
        else:
            print(f"返回结果格式不符合预期: {type(result)}")
            print(f"结果内容: {result}")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        logger.error(f"测试失败: {str(e)}")


def interactive_mode():
    """交互式测试模式"""
    print("\n" + "="*50)
    print("图片生成测试 - 交互式模式")
    print("="*50)
    print("请输入您想要生成的图片描述，然后按回车键。")
    print("使用Gemini模型进行图片生成。")
    print("提示：描述越详细越好，建议包含图片内容、风格等。")
    print("特殊命令:")
    print("  - 输入 'exit' 或 'quit' 退出")
    print("  - 使用 'size:宽x高' 设置图片尺寸 (例如: size:512x512)")
    print("  - 使用 'oss:true' 开启直接返回OSS URL模式")
    print("  - 使用 'oss:false' 关闭直接返回OSS URL模式")
    
    # 默认设置
    current_size = "1024x1024"
    current_oss_url = False
    
    while True:
        # 获取用户输入
        print("\n" + "-"*50)
        print(f"当前设置: 尺寸={current_size}, 直接返回OSS URL={current_oss_url}")
        try:
            user_input = input("请输入图片描述 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n用户中断，退出程序。")
            break
            
        # 检查输入是否为空
        if not user_input:
            print("⚠️ 输入不能为空，请重新输入！")
            continue
            
        # 检查退出条件
        if user_input.lower() in ['exit', 'quit', '退出', '结束']:
            print("退出测试程序。")
            break
            
        # 检查是否为特殊命令
        if user_input.lower().startswith('size:'):
            # 提取尺寸
            size_pattern = r'size:(\d+x\d+)'
            size_match = re.search(size_pattern, user_input.lower())
            if size_match:
                new_size = size_match.group(1)
                current_size = new_size
                print(f"✅ 已设置图片尺寸为: {current_size}")
            else:
                print("⚠️ 尺寸格式错误，应为'size:宽x高'，如'size:512x512'")
            continue
            
        if user_input.lower().startswith('oss:'):
            # 提取OSS设置
            if 'oss:true' in user_input.lower():
                current_oss_url = True
                print("✅ 已开启直接返回OSS URL模式")
            elif 'oss:false' in user_input.lower():
                current_oss_url = False
                print("✅ 已关闭直接返回OSS URL模式")
            else:
                print("⚠️ OSS设置格式错误，应为'oss:true'或'oss:false'")
            continue
        
        # 检查输入有效性
        if len(user_input) < 5:
            print("⚠️ 请输入至少5个字符的描述！")
            continue
        
        # 运行测试
        asyncio.run(test_with_input(user_input, current_size, current_oss_url))


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="图片生成测试工具")
    parser.add_argument(
        "-p", "--prompt", 
        help="直接使用提供的描述进行测试"
    )
    parser.add_argument(
        "-s", "--size",
        default="1024x1024",
        help="图片尺寸，格式为'宽x高'，如'1024x1024'"
    )
    parser.add_argument(
        "--oss",
        action="store_true",
        help="直接返回OSS URL (需要配置OSS)"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="显示更多调试信息"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果设置了debug参数，强制设置更详细的日志级别
    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="TRACE")
        logger.add("tests/image_generator_test.log", rotation="10 MB", level="TRACE")
        logger.info("已启用详细调试模式")
    
    # 打印测试环境信息
    print("="*50)
    print("图片生成测试")
    print("="*50)
    print(f"项目根目录: {project_root}")
    print(f"使用模型: {image_generator_bot.model_id}")
    print(f"图片尺寸: {args.size}")
    print(f"直接返回OSS URL: {args.oss}")
    
    # 检查尺寸格式是否正确
    if not re.match(r'^\d+x\d+$', args.size):
        print(f"错误: 图片尺寸格式错误: {args.size}，应为如 '1024x1024' 的格式")
        return
    
    # 确保结果目录存在
    os.makedirs("tests/results", exist_ok=True)
    
    # 根据传入参数运行测试
    if args.prompt:
        # 确保提示词非空
        if not args.prompt.strip():
            print("错误: 提供的描述不能为空")
            return
            
        # 直接使用提供的描述进行一次测试
        asyncio.run(test_with_input(args.prompt, args.size, args.oss))
    else:
        # 启动交互式测试
        interactive_mode()
    
    print("\n" + "="*50)
    print("测试完成!")
    print("="*50)


if __name__ == "__main__":
    # 运行主函数
    main() 