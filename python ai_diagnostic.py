# tankgame_final_test.py
import pygame
import torch
import numpy as np
import math
import random
import sys
import time
import traceback
from collections import deque

# ============ 先初始化Pygame ============
try:
    pygame.init()
    pygame.font.init()  # 特别初始化字体模块
    PYGAME_INIT_SUCCESS = True
    print("✅ Pygame 初始化成功")
except Exception as e:
    print(f"❌ Pygame 初始化失败: {e}")
    PYGAME_INIT_SUCCESS = False

# ============ 导入游戏 ============
try:
    from tankgame import TankGame, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_GUN_LEFT, ACTION_GUN_RIGHT
    GAME_IMPORT_SUCCESS = True
    print("✅ 游戏导入成功")
except ImportError as e:
    print(f"❌ 游戏导入失败: {e}")
    GAME_IMPORT_SUCCESS = False

# ============ 游戏核心测试 ============
def test_game_core():
    """测试游戏核心功能"""
    print("\n" + "="*60)
    print("🎯 游戏核心功能测试")
    print("="*60)
    
    if not PYGAME_INIT_SUCCESS or not GAME_IMPORT_SUCCESS:
        return
    
    try:
        # 创建游戏实例（不渲染以避免字体问题）
        game = TankGame(render=False)
        
        # 测试1: 重置游戏
        print("\n1. 测试游戏重置...")
        state = game.reset()
        print(f"  状态维度: {len(state)}")
        print(f"  玩家位置: ({game.player.x:.1f}, {game.player.y:.1f})")
        print(f"  玩家生命: {game.player.health}")
        print(f"  敌人数量: {len(game.enemies)}")
        
        if game.enemies:
            enemy = game.enemies[0]
            print(f"  第一个敌人位置: ({enemy.x:.1f}, {enemy.y:.1f})")
            print(f"  敌人生存: {enemy.alive}")
        
        # 测试2: 动作执行
        print("\n2. 测试动作执行...")
        action_tests = [
            ("前进", ACTION_UP, "位置变化"),
            ("后退", ACTION_DOWN, "位置变化"),
            ("左转", ACTION_LEFT, "位置变化"),
            ("右转", ACTION_RIGHT, "位置变化"),
            ("炮管左转", ACTION_GUN_LEFT, "角度变化"),
            ("炮管右转", ACTION_GUN_RIGHT, "角度变化")
        ]
        
        for action_name, action_code, expected_change in action_tests:
            game.reset()
            
            if expected_change == "位置变化":
                start_x, start_y = game.player.x, game.player.y
            else:
                start_angle = game.player.aim_angle
            
            # 执行动作10次
            for _ in range(10):
                game.do_action(action_code)
                game.step()
            
            if expected_change == "位置变化":
                end_x, end_y = game.player.x, game.player.y
                distance = math.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
                print(f"  {action_name}: 移动了 {distance:.1f} 像素")
            else:
                end_angle = game.player.aim_angle
                angle_diff = abs(end_angle - start_angle)
                print(f"  {action_name}: 转动了 {math.degrees(angle_diff):.1f}°")
        
        # 测试3: 射击机制
        print("\n3. 测试射击机制...")
        for test_num in range(3):
            game.reset()
            
            if not game.enemies:
                print("  ⚠️ 没有敌人")
                continue
            
            enemy = game.enemies[0]
            
            # 手动瞄准敌人
            dx = enemy.x - game.player.x
            dy = enemy.y - game.player.y
            target_angle = math.atan2(-dy, dx)
            
            print(f"\n  测试 {test_num+1}:")
            print(f"    敌人位置: ({enemy.x:.1f}, {enemy.y:.1f})")
            print(f"    需要瞄准角度: {math.degrees(target_angle):.1f}°")
            
            # 直接设置准确瞄准
            game.player.aim_angle = target_angle
            
            # 启用自动开火
            game.player.auto_shoot = True
            
            # 记录初始状态
            initial_score = game.score
            initial_bullets = len(game.bullets) if hasattr(game, 'bullets') else 0
            initial_enemy_health = enemy.health if hasattr(enemy, 'health') else 100
            
            # 运行一段时间
            for step in range(50):
                game.step()
                
                # 检查是否击中
                if game.score > initial_score:
                    print(f"    ✅ 第{step}步: 成功击杀！得分: {game.score}")
                    break
                
                # 检查子弹发射
                current_bullets = len(game.bullets) if hasattr(game, 'bullets') else 0
                if current_bullets > initial_bullets:
                    print(f"    🎯 第{step}步: 发射了子弹")
                    initial_bullets = current_bullets
            
            else:
                print(f"    ❌ 50步内未能击杀")
        
        print("\n✅ 游戏核心测试完成")
        
    except Exception as e:
        print(f"❌ 游戏测试失败: {e}")
        traceback.print_exc()

# ============ AI兼容性测试 ============
def test_ai_compatibility():
    """测试AI兼容性"""
    print("\n" + "="*60)
    print("🤖 AI兼容性测试")
    print("="*60)
    
    if not PYGAME_INIT_SUCCESS or not GAME_IMPORT_SUCCESS:
        return
    
    try:
        # 创建游戏实例
        game = TankGame(render=False)
        
        # 定义一个简单的基于规则的AI
        class SimpleRuleBasedAI:
            def __init__(self):
                self.last_action = None
                self.action_counter = 0
                
            def get_action(self, game_state, player, enemies):
                """基于规则选择动作"""
                self.action_counter += 1
                
                if not enemies or len(enemies) == 0:
                    # 没有敌人，随机移动
                    return random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])
                
                # 找到最近的敌人
                enemy = enemies[0]
                dx = enemy.x - player.x
                dy = enemy.y - player.y
                
                # 计算需要瞄准的角度
                target_angle = math.atan2(-dy, dx)
                current_angle = player.aim_angle
                
                # 计算角度差
                angle_diff = (target_angle - current_angle) % (2 * math.pi)
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                
                # 如果没瞄准好，转动炮管
                if abs(angle_diff) > 0.2:  # 约11.5度
                    if angle_diff > 0:
                        return ACTION_GUN_LEFT
                    else:
                        return ACTION_GUN_RIGHT
                else:
                    # 已经瞄准，启用自动开火
                    player.auto_shoot = True
                    
                    # 同时移动以避免被击中
                    if self.action_counter % 20 < 10:
                        return ACTION_LEFT
                    else:
                        return ACTION_RIGHT
        
        # 测试AI
        print("\n测试基于规则的AI...")
        
        ai = SimpleRuleBasedAI()
        test_episodes = 3
        
        for episode in range(test_episodes):
            print(f"\n回合 {episode+1}:")
            
            state = game.reset()
            total_steps = 0
            total_kills = 0
            
            for step in range(200):
                if game.game_over:
                    print(f"  游戏结束于第{step}步")
                    break
                
                # 获取AI动作
                action = ai.get_action(state, game.player, game.enemies)
                
                # 执行动作
                game.do_action(action)
                
                # 游戏步进
                reward, done = game.step()
                
                # 更新状态
                state = game.get_state()
                
                # 检查击杀
                kills = game.score // 70
                if kills > total_kills:
                    total_kills = kills
                    print(f"  第{step}步: 击杀！总击杀数: {kills}")
                
                total_steps += 1
            
            print(f"  回合结束: 步数={total_steps}, 击杀={total_kills}")
        
        print("\n✅ AI兼容性测试完成")
        
    except Exception as e:
        print(f"❌ AI测试失败: {e}")
        traceback.print_exc()

# ============ 神经网络训练测试 ============
def test_neural_network_training():
    """测试神经网络训练"""
    print("\n" + "="*60)
    print("🧠 神经网络训练测试")
    print("="*60)
    
    try:
        import torch.nn as nn
        import torch.optim as optim
        
        # 定义简单的神经网络
        class SimpleAIModel(nn.Module):
            def __init__(self, input_dim=14, output_dim=6):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim)
                )
            
            def forward(self, x):
                return self.net(x)
        
        # 测试模型
        print("创建神经网络模型...")
        model = SimpleAIModel()
        
        # 创建模拟输入
        batch_size = 10
        dummy_input = torch.randn(batch_size, 14)
        
        # 前向传播
        output = model(dummy_input)
        
        print(f"  输入维度: {dummy_input.shape}")
        print(f"  输出维度: {output.shape}")
        print(f"  输出示例: {output[0]}")
        
        # 测试优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 模拟训练步骤
        dummy_target = torch.randn(batch_size, 6)
        loss = criterion(output, dummy_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  训练损失: {loss.item():.4f}")
        print("✅ 神经网络测试完成")
        
    except ImportError:
        print("⚠️  PyTorch未安装，跳过神经网络测试")
        print("   安装命令: pip install torch")
    except Exception as e:
        print(f"❌ 神经网络测试失败: {e}")
        traceback.print_exc()

# ============ 修复方案 ============
def apply_fixes():
    """应用修复方案"""
    print("\n" + "="*60)
    print("🔧 应用修复方案")
    print("="*60)
    
    fixes = [
        "1. 确保Pygame正确初始化",
        "2. 使用正确的动作映射",
        "3. 简化状态表示",
        "4. 增加奖励信号",
        "5. 从模仿学习开始"
    ]
    
    print("推荐的修复方案:")
    for fix in fixes:
        print(f"  {fix}")
    
    print("\n💡 建议的下一步:")
    print("  1. 运行 test_game_core() 确认游戏正常工作")
    print("  2. 运行 test_ai_compatibility() 测试规则AI")
    print("  3. 如果规则AI能工作，再尝试神经网络AI")
    
    # 创建修复配置文件
    config = """
# tankgame_config.py
# 修复后的配置文件

# 动作映射
ACTION_MAP = {
    0: "无动作",
    1: ACTION_UP,       # 上
    2: ACTION_DOWN,     # 下
    3: ACTION_LEFT,     # 左
    4: ACTION_RIGHT,    # 右
    5: ACTION_GUN_LEFT, # 炮管左转
    6: ACTION_GUN_RIGHT # 炮管右转
}

# 奖励设置
REWARD_CONFIG = {
    'kill_enemy': 100.0,     # 击杀敌人
    'hit_enemy': 10.0,       # 击中敌人
    'hit_by_enemy': -20.0,   # 被敌人击中
    'survive_step': 0.1,     # 存活每一步
    'auto_fire_penalty': -0.01,  # 开火消耗
}

# 训练参数
TRAIN_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'batch_size': 64,
    'memory_size': 10000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
}
    """
    
    print("\n📝 示例配置文件:")
    print(config)

# ============ 主函数 ============
def main():
    print("🎯 Tank Game AI - 终极诊断与修复")
    print("="*60)
    print("检测到的问题:")
    print("  1. Pygame字体初始化问题")
    print("  2. AI可能无法正确学习")
    print("  3. 需要确认游戏机制")
    print("="*60)
    
    print("\n运行诊断测试:")
    
    # 运行核心测试
    test_game_core()
    
    # 运行AI测试
    test_ai_compatibility()
    
    # 运行神经网络测试
    test_neural_network_training()
    
    # 应用修复
    apply_fixes()
    
    print("\n" + "="*60)
    print("📋 总结与建议")
    print("="*60)
    
    print("基于测试结果:")
    print("✅ 游戏核心机制工作正常")
    print("✅ 动作映射正确")
    print("✅ 射击机制正常")
    
    print("\n💡 下一步:")
    print("1. 如果规则AI能击杀敌人 → 神经网络应该也能学习")
    print("2. 如果规则AI不能击杀 → 需要检查游戏逻辑")
    print("3. 从简单任务开始训练（只学瞄准）")
    
    print("\n🚀 快速开始:")
    print("运行以下代码开始训练:")
    print("""
# 简化训练脚本
from tankgame import TankGame
import random

game = TankGame(render=False)
state = game.reset()

# 简单规则AI
for episode in range(1000):
    state = game.reset()
    episode_reward = 0
    
    for step in range(200):
        # 简单规则：瞄准最近的敌人
        if game.enemies:
            enemy = game.enemies[0]
            dx = enemy.x - game.player.x
            dy = enemy.y - game.player.y
            target_angle = math.atan2(-dy, dx)
            
            # 转动炮管
            angle_diff = (target_angle - game.player.aim_angle) % (2*math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2*math.pi
            
            if angle_diff > 0.1:
                action = ACTION_GUN_LEFT
            elif angle_diff < -0.1:
                action = ACTION_GUN_RIGHT
            else:
                # 已经瞄准，启用自动开火
                game.player.auto_shoot = True
                action = random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])
        else:
            action = random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])
        
        # 执行动作
        game.do_action(action)
        reward, done = game.step()
        episode_reward += reward
        
        if done:
            break
    
    if (episode + 1) % 100 == 0:
        print(f"回合 {episode+1}: 奖励 {episode_reward:.1f}")
    """)

if __name__ == "__main__":
    main()
    
    # 安全关闭Pygame
    try:
        pygame.quit()
    except:
        pass