#!/usr/bin/env python3
"""
坦克游戏AI观看脚本
加载训练好的模型，观看AI玩游戏
"""

import torch
import numpy as np
import tankgame
import time
import argparse

# 导入训练脚本中的模型和智能体
from train_quick_test import CompactRobustNet, CompactRobustAgent, MOVEMENT_ACTIONS, AIM_ACTIONS

class DemoAgent:
    """演示智能体 - 加载训练好的模型进行游戏演示"""
    
    def __init__(self, model_path="compact_robust_model.pth"):
        self.model = CompactRobustNet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()  # 设置为评估模式
        
        print(f"✓ 已加载模型: {model_path}")
        print(f"✓ 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_action(self, state):
        """根据状态获取动作（无探索，纯贪心）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            movement_q, aim_q = self.model(state_tensor)
            movement_action = movement_q.argmax().item()
            aim_action = aim_q.argmax().item()
        
        return movement_action, aim_action
    
    def get_combined_action(self, movement_action, aim_action):
        """将分离的动作转换为游戏动作列表"""
        actions = []
        if movement_action in MOVEMENT_ACTIONS:
            actions.append(MOVEMENT_ACTIONS[movement_action])
        if aim_action in AIM_ACTIONS:
            actions.append(AIM_ACTIONS[aim_action])
        return actions

def print_game_info():
    """打印游戏控制信息"""
    print("=" * 60)
    print("🎮 坦克游戏AI观看模式")
    print("=" * 60)
    print("控制说明:")
    print("  - 空格键: 暂停/继续")
    print("  - R键: 重新开始")
    print("  - Q键: 退出")
    print("  - H键: 显示/隐藏AI思考信息")
    print()
    print("游戏信息:")
    print("  - 绿色坦克: AI控制的玩家")
    print("  - 蓝色坦克: 敌方AI")
    print("  - 黄色圆点: 玩家子弹")
    print("  - 红色圆点: 敌方子弹")
    print("=" * 60)
    print()

def analyze_ai_thinking(agent, state, movement_action, aim_action):
    """分析AI的思考过程"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    with torch.no_grad():
        movement_q, aim_q = self.model(state_tensor)
    
    # 解析动作名称
    movement_names = ["静止", "向上", "向下", "向左", "向右"]
    aim_names = ["炮管左转", "炮管右转", "射击"]
    
    print(f"🤖 AI思考分析:")
    print(f"   移动Q值: {[f'{q:.2f}' for q in movement_q[0].tolist()]}")
    print(f"   瞄准Q值: {[f'{q:.2f}' for q in aim_q[0].tolist()]}")
    print(f"   选择的移动: {movement_names[movement_action]} (Q值={movement_q[0][movement_action]:.2f})")
    print(f"   选择的瞄准: {aim_names[aim_action]} (Q值={aim_q[0][aim_action]:.2f})")
    
    # 分析游戏状态
    player_x = state[0] * tankgame.SCREEN_WIDTH
    player_y = state[1] * tankgame.SCREEN_HEIGHT
    player_lives = state[3] * 5
    
    if state[11] > 0.5:  # 有敌人
        enemy_x = state[5] * tankgame.SCREEN_WIDTH
        enemy_y = state[6] * tankgame.SCREEN_HEIGHT
        distance = state[7] * tankgame.get_screen_diag()
        print(f"   战术分析: 玩家位置({player_x:.0f},{player_y:.0f}), "
              f"敌人位置({enemy_x:.0f},{enemy_y:.0f}), 距离{distance:.0f}")
    
    print(f"   生存状态: 生命值{player_lives:.0f}, "
          f"时间剩余{state[32] * tankgame.GAME_TIME_LIMIT:.0f}秒")
    print("-" * 40)

def watch_ai_play(args):
    """观看AI玩游戏的主函数"""
    
    # 打印游戏信息
    print_game_info()
    
    # 创建游戏和智能体
    game = tankgame.TankGame(render=True)
    agent = DemoAgent(args.model)
    
    # 游戏统计
    episode_count = 0
    total_score = 0
    best_score = 0
    
    # 控制变量
    paused = False
    show_thinking = args.verbose
    auto_restart = args.auto_restart
    
    print(f"🎯 开始观看AI游戏...")
    print(f"   自动重启: {'开启' if auto_restart else '关闭'}")
    print(f"   显示思考: {'开启' if show_thinking else '关闭'}")
    print()
    
    try:
        while True:
            episode_count += 1
            state = game.reset()
            episode_score = 0
            step_count = 0
            
            print(f"📍 第 {episode_count} 局开始")
            
            # 游戏主循环
            while True:
                # 处理事件
                for event in tankgame.pygame.event.get():
                    if event.type == tankgame.pygame.QUIT:
                        print("👋 退出观看")
                        return
                    elif event.type == tankgame.pygame.KEYDOWN:
                        if event.key == tankgame.pygame.K_q:
                            print("👋 退出观看")
                            return
                        elif event.key == tankgame.pygame.K_SPACE:
                            paused = not paused
                            print(f"⏸️  {'暂停' if paused else '继续'}")
                        elif event.key == tankgame.pygame.K_r:
                            print("🔄 手动重新开始")
                            game.reset()
                            state = game.get_state()
                        elif event.key == tankgame.pygame.K_h:
                            show_thinking = not show_thinking
                            print(f"💭 思考信息: {'显示' if show_thinking else '隐藏'}")
                
                if paused:
                    tankgame.pygame.time.wait(100)
                    continue
                
                # AI决策
                movement_action, aim_action = agent.get_action(state)
                actions = agent.get_combined_action(movement_action, aim_action)
                
                # 显示AI思考过程
                if show_thinking and step_count % 30 == 0:  # 每30帧显示一次
                    analyze_ai_thinking(agent, state, movement_action, aim_action)
                
                # 执行动作
                game.do_actions(actions)
                reward, done = game.step()
                next_state = game.get_state()
                
                episode_score += reward
                step_count += 1
                state = next_state
                
                # 游戏结束处理
                if done:
                    total_score += game.score
                    if game.score > best_score:
                        best_score = game.score
                    
                    print(f"🏁 第 {episode_count} 局结束")
                    print(f"   游戏分数: {game.score}")
                    print(f"   回合奖励: {episode_score:.1f}")
                    print(f"   总步数: {step_count}")
                    print(f"   历史最佳: {best_score}")
                    print(f"   平均分数: {total_score/episode_count:.1f}")
                    print()
                    
                    # 自动重启或等待
                    if auto_restart:
                        tankgame.pygame.time.wait(2000)  # 等待2秒
                        break
                    else:
                        print("按R键重新开始，Q键退出")
                        # 等待用户输入
                        waiting = True
                        while waiting:
                            for event in tankgame.pygame.event.get():
                                if event.type == tankgame.pygame.QUIT:
                                    return
                                elif event.type == tankgame.pygame.KEYDOWN:
                                    if event.key == tankgame.pygame.K_q:
                                        return
                                    elif event.key == tankgame.pygame.K_r:
                                        waiting = False
                                        break
                            tankgame.pygame.time.wait(100)
                        break
    
    except KeyboardInterrupt:
        print("\n👋 观看被中断")
    
    # 最终统计
    print("=" * 60)
    print("📊 观看统计:")
    print(f"   总局数: {episode_count}")
    print(f"   平均分数: {total_score/episode_count:.1f}")
    print(f"   最佳分数: {best_score}")
    print("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="观看训练好的AI玩坦克游戏")
    parser.add_argument("--model", default="compact_robust_model.pth", 
                       help="模型文件路径 (默认: compact_robust_model.pth)")
    parser.add_argument("--auto-restart", action="store_true",
                       help="游戏结束后自动重新开始")
    parser.add_argument("--verbose", action="store_true",
                       help="显示AI的详细思考过程")
    
    args = parser.parse_args()
    
    # 检查模型文件
    try:
        torch.load(args.model, map_location='cpu')
    except FileNotFoundError:
        print(f"❌ 找不到模型文件: {args.model}")
        print("请确保模型文件存在，或使用 --model 参数指定正确的路径")
        return
    
    # 开始观看
    watch_ai_play(args)

if __name__ == "__main__":
    main()