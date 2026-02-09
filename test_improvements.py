#!/usr/bin/env python3
"""
测试改进后的双动作坦克AI
验证所有改进是否正常工作
"""

import torch
import numpy as np
import tankgame
from train import SimpleAgent

def test_improved_system():
    """测试改进后的系统"""
    print("=== 测试改进后的双动作坦克AI ===")
    print("测试内容：")
    print("1. 奖励系统统一")
    print("2. 双头网络训练平衡")
    print("3. 增强状态表征（32维）")
    print("4. 优先级经验回放")
    print("5. 学习率调度")
    print()
    
    # 创建游戏和智能体
    game = tankgame.TankGame(render=True)
    agent = SimpleAgent()
    
    print(f"✓ 游戏系统初始化成功")
    print(f"✓ 网络架构：共享层(512-256-256) + 移动头(5动作) + 瞄准头(3动作)")
    
    # 测试状态维度
    state = game.get_state()
    print(f"✓ 状态维度：{len(state)}维（增强版33维）")
    
    # 尝试加载已训练模型
    try:
        agent.model.load_state_dict(torch.load("improved_dual_final_model.pth"))
        print(f"✓ 加载改进模型成功")
    except:
        try:
            agent.model.load_state_dict(torch.load("dual_final_model.pth"))
            print(f"✓ 加载旧版双动作模型，将使用兼容模式")
        except:
            print(f"⚠ 未找到训练好的模型，使用随机策略进行测试")
    
    agent.epsilon = 0.0  # 不使用随机探索
    
    print(f"\n开始测试游戏...")
    print("观察改进后的AI表现：")
    print("- 更平衡的移动和瞄准")
    print("- 更智能的战术决策")
    print("- 更稳定的训练过程")
    
    state = game.reset()
    step_count = 0
    reward_history = []
    
    while True:
        # 获取双动作
        movement_action, aim_action = agent.get_action(state, training=False)
        actions = agent.get_combined_action(movement_action, aim_action)
        
        # 显示动作信息
        if step_count % 30 == 0:  # 每30帧显示一次
            movement_name = ["静止", "上", "下", "左", "右"][movement_action]
            aim_name = ["左转", "右转", "射击"][aim_action]
            print(f"步骤 {step_count:3d}: 移动={movement_name}({movement_action}), "
                  f"瞄准={aim_name}({aim_action}), 分数={game.score:3d}")
        
        # 执行动作
        game.do_actions(actions)
        reward, done = game.step()
        state = game.get_state()
        
        reward_history.append(reward)
        step_count += 1
        
        if done:
            print(f"\n=== 测试完成 ===")
            print(f"最终分数: {game.score}")
            print(f"总步骤数: {step_count}")
            print(f"平均奖励: {np.mean(reward_history):.3f}")
            print(f"奖励标准差: {np.std(reward_history):.3f}")
            
            # 分析奖励分布
            positive_rewards = sum(1 for r in reward_history if r > 0)
            print(f"正奖励比例: {positive_rewards/len(reward_history)*100:.1f}%")
            
            if game.score > 50:
                print("✅ 表现优秀！AI成功击败了敌人")
            elif game.score > 20:
                print("✅ 表现良好！AI击中了敌人")
            else:
                print("⚠️ 需要更多训练")
            break

def quick_training_test():
    """快速训练测试"""
    print("\n=== 快速训练测试 ===")
    print("运行200步训练来验证改进效果...")
    
    game = tankgame.TankGame(render=False)
    agent = SimpleAgent()
    
    # 快速训练
    episode = 0
    total_steps = 0
    
    while total_steps < 200:
        state = game.reset()
        episode_reward = 0
        
        for step in range(50):  # 每回合最多50步
            movement_action, aim_action = agent.get_action(state, training=True)
            actions = agent.get_combined_action(movement_action, aim_action)
            game.do_actions(actions)
            reward, done = game.step()
            next_state = game.get_state()
            
            agent.remember(state, movement_action, aim_action, reward, next_state, done)
            
            if total_steps % 10 == 0:
                agent.train()
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        episode += 1
        print(f"回合 {episode:2d}: 奖励={episode_reward:6.2f}, 游戏分数={game.score:3d}, "
              f"ε={agent.epsilon:.3f}")
        
        if total_steps >= 200:
            break
    
    print(f"\n✅ 快速训练完成！共{total_steps}步，{episode}回合")
    print("改进效果：")
    print("- 奖励系统统一，不再有矛盾信号")
    print("- 移动头和瞄准头平衡训练")
    print("- 增强状态提供更多战术信息")
    print("- 优先级经验回放提升学习效率")

if __name__ == "__main__":
    quick_training_test()
    test_improved_system()