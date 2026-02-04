# fix_and_test.py
import pygame
import torch
import numpy as np
import math
import random

# 导入游戏
from tankgame import TankGame, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_GUN_LEFT, ACTION_GUN_RIGHT

def test_action_mapping():
    """测试动作映射是否正确"""
    print("\n🔍 测试动作映射...")
    
    # 显示游戏中的动作常量
    print("游戏中的动作常量:")
    print(f"  ACTION_UP = {ACTION_UP}")
    print(f"  ACTION_DOWN = {ACTION_DOWN}")
    print(f"  ACTION_LEFT = {ACTION_LEFT}")
    print(f"  ACTION_RIGHT = {ACTION_RIGHT}")
    print(f"  ACTION_GUN_LEFT = {ACTION_GUN_LEFT}")
    print(f"  ACTION_GUN_RIGHT = {ACTION_GUN_RIGHT}")
    
    # 创建游戏测试
    game = TankGame(render=True)
    game.reset()
    
    # 测试每个动作
    actions = [
        ("上", ACTION_UP),
        ("下", ACTION_DOWN),
        ("左", ACTION_LEFT),
        ("右", ACTION_RIGHT),
        ("炮左转", ACTION_GUN_LEFT),
        ("炮右转", ACTION_GUN_RIGHT)
    ]
    
    for action_name, action_code in actions:
        initial_pos = (game.player.x, game.player.y)
        initial_angle = game.player.aim_angle
        
        # 执行动作
        game.do_action(action_code)
        game.step()
        
        final_pos = (game.player.x, game.player.y)
        final_angle = game.player.aim_angle
        
        if action_name in ["上", "下", "左", "右"]:
            moved = (abs(final_pos[0] - initial_pos[0]) > 1 or 
                    abs(final_pos[1] - initial_pos[1]) > 1)
            print(f"  {action_name}({action_code}): 移动={moved}")
        else:
            angle_changed = abs(final_angle - initial_angle) > 0.01
            print(f"  {action_name}({action_code}): 角度变化={angle_changed}")
    
    pygame.quit()

def test_auto_shoot():
    """测试自动开火机制"""
    print("\n🔫 测试自动开火机制...")
    
    game = TankGame(render=False)
    game.reset()
    
    # 获取初始子弹数量
    initial_bullets = len(game.bullets)
    
    # 启用自动开火
    game.player.auto_shoot = True
    
    # 瞄准敌人
    if game.enemies:
        enemy = game.enemies[0]
        dx = enemy.x - game.player.x
        dy = enemy.y - game.player.y
        game.player.aim_angle = math.atan2(-dy, dx)
    
    # 运行几步
    for _ in range(30):
        game.step()
    
    final_bullets = len(game.bullets)
    
    print(f"  初始子弹: {initial_bullets}")
    print(f"  最终子弹: {final_bullets}")
    print(f"  发射子弹: {final_bullets - initial_bullets}")
    
    if final_bullets > initial_bullets:
        print("  ✅ 自动开火正常工作")
    else:
        print("  ❌ 自动开火可能有问题")
    
    pygame.quit()

def test_state_dimension():
    """测试状态维度"""
    print("\n📊 测试状态维度...")
    
    game = TankGame(render=False)
    state = game.reset()
    
    print(f"  状态长度: {len(state)}")
    print(f"  状态值: {state}")
    
    # 检查是否有NaN或异常值
    has_nan = any(np.isnan(x) for x in state)
    has_inf = any(np.isinf(x) for x in state)
    
    if has_nan:
        print("  ❌ 状态包含NaN值")
    if has_inf:
        print("  ❌ 状态包含Inf值")
    
    if not has_nan and not has_inf:
        print("  ✅ 状态表示正常")
    
    pygame.quit()

def create_compatible_ai():
    """创建与游戏完全兼容的AI"""
    print("\n🤖 创建兼容AI...")
    
    # 根据游戏的动作常量重新定义
    GAME_ACTIONS = {
        0: "无动作",
        1: ACTION_UP,      # 上
        2: ACTION_DOWN,    # 下
        3: ACTION_LEFT,    # 左
        4: ACTION_RIGHT,   # 右
        5: ACTION_GUN_LEFT, # 炮左转
        6: ACTION_GUN_RIGHT # 炮右转
    }
    
    class CompatibleAI:
        def __init__(self):
            # 简单规则：总是瞄准并射击
            self.last_enemy_angle = 0
            
        def get_action(self, game_state):
            """根据状态返回动作"""
            # state[2] 是玩家当前角度，state[6] 是敌人方向角度
            current_angle = game_state[2] * 2 * math.pi  # 反归一化
            enemy_angle = game_state[6] * 2 * math.pi if game_state[6] > 0 else current_angle
            
            # 计算角度差
            angle_diff = (enemy_angle - current_angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            
            # 决策：如果没瞄准好，转动炮管
            if abs(angle_diff) > 0.1:  # 10%误差
                if angle_diff > 0:
                    return 5  # 炮左转
                else:
                    return 6  # 炮右转
            else:
                # 已经瞄准，随机移动
                return random.choice([1, 2, 3, 4])
    
    return CompatibleAI()

def test_compatible_ai():
    """测试兼容AI"""
    print("\n🧪 测试兼容AI...")
    
    game = TankGame(render=True)
    ai = create_compatible_ai()
    
    total_kills = 0
    test_episodes = 5
    
    for ep in range(test_episodes):
        state = game.reset()
        kills = 0
        
        print(f"\n回合 {ep+1}:")
        
        for step in range(200):
            if game.game_over:
                break
            
            # 获取AI动作
            action = ai.get_action(state)
            
            # 执行动作
            game.do_action(action)
            
            # 启用自动开火（当瞄准较好时）
            enemy_angle = state[6] * 2 * math.pi if state[6] > 0 else 0
            current_angle = state[2] * 2 * math.pi
            angle_diff = abs(enemy_angle - current_angle) % (2 * math.pi)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            
            # 如果瞄准误差小于20%，自动开火
            game.player.auto_shoot = (angle_diff / math.pi) < 0.2
            
            # 游戏步进
            game.step()
            
            # 更新状态
            state = game.get_state()
            
            # 检查击杀
            current_kills = game.score // 70
            if current_kills > kills:
                kills = current_kills
                print(f"  步{step}: 击杀！总击杀{kills}")
            
            # 检查退出
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        total_kills += kills
        print(f"  回合结束: 击杀{kills}")
    
    pygame.quit()
    
    avg_kills = total_kills / test_episodes
    print(f"\n📊 兼容AI测试: 平均每回合{avg_kills:.1f}击杀")
    
    if avg_kills > 0:
        print("✅ 兼容AI能成功击杀！")
        print("💡 问题可能是AI训练时的动作映射错误")
    else:
        print("❌ 即使兼容AI也无法击杀，可能是游戏机制问题")

def quick_fix_training():
    """快速修复训练：使用正确的动作映射"""
    print("\n⚡ 快速修复训练...")
    
    import torch.nn as nn
    import torch.optim as optim
    
    # 根据游戏动作重新定义
    STATE_DIM = 14
    ACTION_DIM = 7  # 0-6，但0是ACTION_IDLE，我们只用1-6
    
    class FixedActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, 32),
                nn.ReLU(),
                nn.Linear(32, ACTION_DIM)
            )
            
            # 初始化偏向瞄准动作(5,6)
            with torch.no_grad():
                self.net[-1].weight[5] += 0.3  # 炮左转
                self.net[-1].weight[6] += 0.3  # 炮右转
                self.net[-1].bias[5] += 0.2
                self.net[-1].bias[6] += 0.2
        
        def forward(self, x):
            return nn.functional.softmax(self.net(x), dim=-1)
    
    # 动作映射
    ACTION_MAP = {
        1: ACTION_UP,
        2: ACTION_DOWN,
        3: ACTION_LEFT,
        4: ACTION_RIGHT,
        5: ACTION_GUN_LEFT,
        6: ACTION_GUN_RIGHT
    }
    
    # 训练
    game = TankGame(render=False)
    model = FixedActor()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("训练100轮...")
    
    for episode in range(100):
        state = game.reset()
        episode_reward = 0
        episode_kills = 0
        
        for step in range(100):
            if game.game_over:
                break
            
            # 状态转tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 获取动作概率（排除ACTION_IDLE=0）
            with torch.no_grad():
                probs = model(state_tensor)[0]
                action_probs = probs[1:]  # 只用1-6
                action_idx = torch.multinomial(action_probs, 1).item() + 1
            
            # 执行动作
            game.do_action(ACTION_MAP[action_idx])
            
            # 自动开火逻辑
            if action_idx in [5, 6]:  # 瞄准动作
                # 检查瞄准误差
                if state[6] > 0:  # 有敌人
                    enemy_angle = state[6] * 2 * math.pi
                    current_angle = state[2] * 2 * math.pi
                    angle_diff = abs(enemy_angle - current_angle) % (2 * math.pi)
                    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                    
                    if angle_diff / math.pi < 0.3:  # 30%误差内
                        game.player.auto_shoot = True
            
            # 游戏步进
            reward, done = game.step()
            episode_reward += reward
            
            # 更新状态
            state = game.get_state()
            
            # 检查击杀
            current_kills = game.score // 70
            if current_kills > episode_kills:
                episode_kills = current_kills
        
        if (episode + 1) % 20 == 0:
            print(f"  轮次 {episode+1}: 奖励{episode_reward:.1f}, 击杀{episode_kills}")
    
    pygame.quit()
    
    # 保存模型
    torch.save(model.state_dict(), "./fixed_ai_model.pth")
    print("💾 修复模型已保存: fixed_ai_model.pth")

def main():
    print("🎯 Tank Game AI 兼容性修复")
    print("=" * 60)
    
    print("运行诊断测试...")
    
    # 1. 测试动作映射
    test_action_mapping()
    
    # 2. 测试自动开火
    test_auto_shoot()
    
    # 3. 测试状态维度
    test_state_dimension()
    
    # 4. 测试兼容AI
    test_compatible_ai()
    
    # 5. 提供修复选项
    print("\n" + "=" * 60)
    print("修复选项:")
    print("1. 运行快速修复训练")
    print("2. 修改AI训练脚本使用正确动作映射")
    print("3. 退出")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        quick_fix_training()
    elif choice == "2":
        print("\n💡 需要修改AI训练脚本:")
        print("将AI中的动作映射改为:")
        print("  ACTION_UP = 1")
        print("  ACTION_DOWN = 2")
        print("  ACTION_LEFT = 3")
        print("  ACTION_RIGHT = 4")
        print("  ACTION_GUN_LEFT = 5")
        print("  ACTION_GUN_RIGHT = 6")
    else:
        print("👋 退出")

if __name__ == "__main__":
    # 确保PyTorch可用
    try:
        import torch
        main()
    except ImportError:
        print("❌ 需要安装PyTorch: pip install torch")
        
        # 只运行不需要PyTorch的测试
        test_action_mapping()
        test_auto_shoot()
        test_state_dimension()
        test_compatible_ai()