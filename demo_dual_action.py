#!/usr/bin/env python3
"""
双动作模型演示脚本
展示新训练的双动作模型如何同时进行移动和瞄准
"""

import torch
import numpy as np
import tankgame
from train import SimpleAgent, MOVEMENT_ACTIONS, AIM_ACTIONS

def demo_dual_action():
    """演示双动作模型的功能"""
    print("=== 双动作坦克AI演示 ===")
    print("模型结构：")
    print("- 移动头：5个动作（静止、上、下、左、右）")
    print("- 瞄准头：3个动作（左转、右转、射击）")
    print("- 可以同时执行移动+瞄准动作\n")
    
    # 创建游戏和智能体
    game = tankgame.TankGame(render=True)
    agent = SimpleAgent()
    
    # 尝试加载训练好的模型
    try:
        agent.model.load_state_dict(torch.load("dual_final_model.pth"))
        print("✓ 加载双动作模型成功")
    except:
        print("✗ 未找到训练好的双动作模型，使用随机策略")
        print("  运行 'python train.py' 来训练新模型")
    
    agent.epsilon = 0.0  # 不使用随机探索
    
    # 开始演示
    print("\n开始游戏演示...")
    print("观察AI如何同时进行移动和瞄准")
    state = game.reset()
    step_count = 0
    
    while True:
        # 获取双动作
        movement_action, aim_action = agent.get_action(state, training=False)
        actions = agent.get_combined_action(movement_action, aim_action)
        
        # 显示动作信息
        if step_count % 30 == 0:  # 每30帧显示一次
            movement_name = ["静止", "上", "下", "左", "右"][movement_action]
            aim_name = ["左转", "右转", "射击"][aim_action]
            print(f"步骤 {step_count}: 移动={movement_name}, 瞄准={aim_name}")
        
        # 执行动作
        game.do_actions(actions)
        reward, done = game.step()
        state = game.get_state()
        step_count += 1
        
        if done:
            print(f"\n游戏结束！")
            print(f"最终分数: {game.score}")
            print(f"总步骤数: {step_count}")
            break

def show_action_mapping():
    """显示动作映射"""
    print("\n=== 动作映射表 ===")
    print("移动动作:")
    for i, action in MOVEMENT_ACTIONS.items():
        names = ["静止", "上", "下", "左", "右"]
        print(f"  {i} -> {names[i]} (游戏动作: {action})")
    
    print("\n瞄准动作:")
    for i, action in AIM_ACTIONS.items():
        names = ["左转", "右转", "射击"]
        print(f"  {i} -> {names[i]} (游戏动作: {action})")

if __name__ == "__main__":
    show_action_mapping()
    demo_dual_action()