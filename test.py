#!/usr/bin/env python3
"""
简单测试脚本 - 用于验证训练后的模型性能
"""
import torch
import torch.nn as nn
import numpy as np
import tankgame
import time

class FinalOptimizedNet(nn.Module):
    """最终优化网络架构"""
    def __init__(self):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(67, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 96),
            nn.ReLU()
        )
        
        self.movement_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 5)
        )
        
        self.aim_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 3)
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        movement_q = self.movement_head(shared_features)
        aim_q = self.aim_head(shared_features)
        return movement_q, aim_q

def test_model(model_path, episodes=20):
    """测试指定模型的性能"""
    print(f"🔍 测试模型: {model_path}")
    
    game = tankgame.TankGame(render=False)
    
    # 加载模型
    model = FinalOptimizedNet()
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"✓ 成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"✗ 模型文件不存在: {model_path}")
        return None, None
    
    scores = []
    game_scores = []
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 纯模型决策，无探索
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                movement_q, aim_q = model(state_tensor)
                movement_action = movement_q.argmax().item()
                aim_action = aim_q.argmax().item()
            
            # 执行动作
            actions = []
            if movement_action == 1:
                actions.append(tankgame.ACTION_UP)
            elif movement_action == 2:
                actions.append(tankgame.ACTION_DOWN)
            elif movement_action == 3:
                actions.append(tankgame.ACTION_LEFT)
            elif movement_action == 4:
                actions.append(tankgame.ACTION_RIGHT)
            
            if aim_action == 0:
                actions.append(tankgame.ACTION_GUN_LEFT)
            elif aim_action == 1:
                actions.append(tankgame.ACTION_GUN_RIGHT)
            elif aim_action == 2:
                actions.append(tankgame.ACTION_SHOOT)
            
            game.do_actions(actions)
            reward, done = game.step()
            next_state = game.get_state()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or steps > 300:
                break
        
        scores.append(total_reward)
        game_scores.append(game.score)
    
    avg_score = np.mean(scores)
    avg_game_score = np.mean(game_scores)
    
    print(f"  📊 结果:")
    print(f"    平均奖励: {avg_score:.1f} ± {np.std(scores):.1f}")
    print(f"    平均游戏分数: {avg_game_score:.1f} ± {np.std(game_scores):.1f}")
    print(f"    最高分数: {max(game_scores)}")
    print(f"    平均击杀数: {avg_game_score/70:.1f}")
    
    return avg_score, avg_game_score

if __name__ == "__main__":
    print("🎯 模型性能测试")
    print("=" * 40)
    
    # 测试可能的模型文件
    models_to_test = [
        "best_model.pth",
        "final_model_1000.pth"
    ]
    
    best_score = 0
    best_model = None
    
    for model_path in models_to_test:
        avg_score, avg_game_score = test_model(model_path, episodes=20)
        if avg_game_score and avg_game_score > best_score:
            best_score = avg_game_score
            best_model = model_path
    
    if best_model:
        print(f"\n🏆 最佳模型: {best_model}")
        print(f"最佳平均分数: {best_score:.1f}")
        
        if best_score > 100:
            rating = "🏆 优秀"
        elif best_score > 50:
            rating = "🥈 良好"
        elif best_score > 0:
            rating = "🥉 及格"
        else:
            rating = "❌ 不及格"
        
        print(f"性能评价: {rating}")
    else:
        print("\n❌ 未找到可用的模型文件")