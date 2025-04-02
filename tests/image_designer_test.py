#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片设计机器人测试脚本 (合并版)

用于测试image_designer.py中的ImageDesignerBot功能
支持控制台输入的交互式测试
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from loguru import logger

# 确保能够导入app模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入被测试的组件
from app.tools.image_designer import ImageDesignerBot, image_designer_bot, image_designer
from app.core.config import settings

# 配置日志
logger.remove()  # 移除默认处理器
logger.add(sys.stderr, level="INFO")  # 添加stderr处理器
logger.add("tests/image_designer_test.log", rotation="10 MB", level="DEBUG")  # 添加文件处理器


async def test_with_input(prompt: str):
    """
    使用用户输入的描述测试图片设计机器人
    
    Args:
        prompt: 用户输入的图片描述
    """
    logger.info(f"使用输入描述测试: {prompt[:50]}...")
    
    try:
        # 检查API密钥
        if not settings.DEEPSEEK_API_KEY:
            print("\n错误: 未配置DeepSeek API密钥，请在.env文件中设置DEEPSEEK_API_KEY")
            print("您可以使用以下格式添加到.env文件：")
            print("DEEPSEEK_API_KEY=your_api_key_here")
            print("DEEPSEEK_API_BASE=https://api.deepseek.com/v1")
            return
        
        # 创建结果目录
        os.makedirs("tests/results", exist_ok=True)
        
        # 显示进度
        print("\n正在生成设计方案，请稍候...\n")
        
        # 调用图片设计机器人 - 使用ainvoke方法
        result = await image_designer.ainvoke({"user_demand": prompt})
        
        # 输出结果
        print("\n" + "="*50)
        print("图片设计结果")
        print("="*50)
        
        # 输出分析结果
        if isinstance(result, dict):
            # 处理分析结果
            if "分析结果" in result:
                print("\n📊 分析结果:")
                for key, value in result["分析结果"].items():
                    print(f"  ▶ {key}: {value}")
            
            # 处理设计方案
            for plan_key in [k for k in result.keys() if "方案" in k]:
                print(f"\n🎨 {plan_key}:")
                plan = result[plan_key]
                
                if isinstance(plan, dict):
                    # 输出方案详情
                    for detail_key, detail_value in plan.items():
                        if detail_key == "标题":
                            print(f"  📌 {detail_key}: {detail_value}")
                        else:
                            print(f"  ▶ {detail_key}: {detail_value}")
                else:
                    print(f"  {plan}")
            
            # 保存详细结果到文件
            timestamp = asyncio.get_event_loop().time()
            result_file = f"tests/results/design_result_{int(timestamp)}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n完整结果已保存至: {result_file}")
        else:
            print(f"返回结果格式不符合预期: {type(result)}")
            print(f"结果内容: {result}")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        logger.error(f"测试失败: {str(e)}")


async def test_with_predefined_cases():
    """使用预定义的测试用例测试图片设计机器人"""
    logger.info("开始测试图片设计机器人...")
    
    # 测试用例列表 - 不同类型的图片描述
    test_cases = [
        # 创意型需求 - 预期使用高温度
        "设计一个梦幻的水下城市，充满奇幻和超现实的元素，有会发光的建筑和奇特的海洋生物",
        
        # 精确型需求 - 预期使用低温度
        "设计一个现代简约风格的客厅，要真实精确，包含白色墙面、灰色沙发、木质茶几和一盆绿植",
        
        # 平衡型需求
        "设计一个秋天的公园场景，有金黄色的树叶和一条小路，阳光透过树叶洒在地上"
    ]
    
    # 检查API密钥是否配置
    if not settings.DEEPSEEK_API_KEY:
        logger.error("未配置DeepSeek API密钥，请在.env文件中设置DEEPSEEK_API_KEY")
        logger.info("您可以使用以下格式添加到.env文件：")
        logger.info("DEEPSEEK_API_KEY=your_api_key_here")
        logger.info("DEEPSEEK_API_BASE=https://api.deepseek.com/v1")
        return
    
    # 使用全局实例进行测试
    try:
        # 创建结果目录
        os.makedirs("tests/results", exist_ok=True)
        
        # 按序测试每个用例
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n测试用例 {i+1}: {test_case[:50]}...")
            print(f"\n正在测试用例 {i+1}/{len(test_cases)}: {test_case}")
            
            try:
                # 调用图片设计机器人 - 使用ainvoke方法
                result = await image_designer.ainvoke({"user_demand": test_case})
                
                # 输出结果概要
                logger.info(f"测试用例 {i+1} 成功!")
                print(f"✅ 测试用例 {i+1} 成功!")
                
                # 格式化输出结果
                if isinstance(result, dict):
                    # 尝试打印分析结果关键部分
                    if "分析结果" in result:
                        for key, value in result["分析结果"].items():
                            logger.info(f"  {key}: {value[:100]}...")
                    
                    # 打印设计方案标题
                    for plan_key in [k for k in result.keys() if "方案" in k]:
                        if "标题" in result[plan_key]:
                            title = result[plan_key]["标题"]
                            logger.info(f"  {plan_key} 标题: {title}")
                            print(f"  🎨 {plan_key} 标题: {title}")
                    
                    # 保存详细结果到文件
                    result_file = f"tests/results/case_{i+1}_result.json"
                    with open(result_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"完整结果已保存至: {result_file}")
                    print(f"  💾 完整结果已保存至: {result_file}")
                else:
                    logger.warning(f"返回结果格式不符合预期: {type(result)}")
                    print(f"  ⚠️ 返回结果格式不符合预期: {type(result)}")
            
            except Exception as e:
                logger.error(f"测试用例 {i+1} 失败: {str(e)}")
                print(f"❌ 测试用例 {i+1} 失败: {str(e)}")
            
            # 在测试用例之间添加延迟，避免API限流
            if i < len(test_cases) - 1:
                print(f"  ⏳ 等待2秒后继续下一个测试...")
                await asyncio.sleep(2)
    
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        print(f"❌ 测试过程中发生错误: {str(e)}")
    
    logger.info("测试完成!")
    print("\n所有测试用例已完成!")


async def test_temperature_adjustment():
    """测试温度调整功能"""
    logger.info("开始测试温度调整功能...")
    print("\n开始测试温度调整功能...")
    
    # 创建测试实例
    try:
        designer = ImageDesignerBot()
        
        # 测试用例 - 预期不同温度结果
        test_cases = [
            ("梦幻创意超现实的外星风景", 0.8),  # 创意型 - 高温度
            ("精确写实的产品照片", 0.4),        # 精确型 - 低温度
            ("山间小屋夕阳风景", 0.7)            # 平衡型 - 中温度
        ]
        
        # 测试每个用例的温度调整
        for prompt, expected_temp in test_cases:
            temp = await designer._adjust_temperature(prompt)
            result = "✓ 通过" if abs(temp - expected_temp) < 0.01 else "✗ 不通过"
            
            logger.info(f"提示词: '{prompt}'")
            logger.info(f"预期温度: {expected_temp}, 实际温度: {temp}")
            logger.info(f"结果: {result}")
            
            print(f"提示词: '{prompt}'")
            print(f"  预期温度: {expected_temp}, 实际温度: {temp}")
            print(f"  结果: {result}")
    
    except Exception as e:
        logger.error(f"温度调整测试失败: {str(e)}")
        print(f"❌ 温度调整测试失败: {str(e)}")


def interactive_mode():
    """交互式测试模式"""
    print("\n" + "="*50)
    print("图片设计机器人 - 交互式测试模式")
    print("="*50)
    print("请输入您想要测试的图片描述，然后按回车键。")
    print("输入 'exit' 或 'quit' 退出。")
    
    while True:
        # 获取用户输入
        print("\n" + "-"*50)
        user_input = input("请输入图片描述 > ").strip()
        
        # 检查退出条件
        if user_input.lower() in ['exit', 'quit', '退出', '结束']:
            print("退出测试程序。")
            break
        
        # 检查输入有效性
        if not user_input or len(user_input) < 5:
            print("⚠️ 请输入至少5个字符的描述！")
            continue
        
        # 运行测试
        asyncio.run(test_with_input(user_input))


def batch_mode():
    """批处理测试模式"""
    print("\n" + "="*50)
    print("图片设计机器人 - 批处理测试模式")
    print("="*50)
    
    # 运行温度调整测试
    asyncio.run(test_temperature_adjustment())
    
    # 运行预定义用例测试
    asyncio.run(test_with_predefined_cases())


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="图片设计机器人测试工具")
    parser.add_argument(
        "-m", "--mode", 
        choices=["interactive", "batch"], 
        default="interactive",
        help="测试模式: interactive (交互式) 或 batch (批处理，使用预定义用例)"
    )
    parser.add_argument(
        "-p", "--prompt", 
        help="直接使用提供的描述进行测试（仅在交互式模式生效）"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 打印测试环境信息
    print("="*50)
    print("图片设计机器人测试")
    print("="*50)
    print(f"项目根目录: {project_root}")
    print(f"Python版本: {sys.version}")
    print(f"运行模式: {args.mode}")
    
    # 根据模式运行测试
    if args.mode == "interactive":
        if args.prompt:
            # 直接使用提供的描述进行一次测试
            asyncio.run(test_with_input(args.prompt))
        else:
            # 启动交互式测试
            interactive_mode()
    else:
        # 批处理模式
        batch_mode()
    
    print("\n" + "="*50)
    print("测试完成!")
    print("="*50)


if __name__ == "__main__":
    # 创建测试目录
    os.makedirs("tests", exist_ok=True)
    
    # 运行主函数
    main() 