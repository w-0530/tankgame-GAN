import os
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from tankgame import (
    TankGame, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_GUN_LEFT, ACTION_GUN_RIGHT
)

# 超级简单的测试脚本
def super_simple_test(model_path):
    """最简单的测试脚本"""
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        # 尝试查找其他模型
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if files:
                model_path = os.path.join(model_dir, files[0])
                print(f"🔍 使用找到的模型: {model_path}")
            else:
                print(f"❌ 没有找到任何模型文件")
                return
        else:
            print(f"❌ 模型目录不存在")
            return
    
    pygame.init()
    game = TankGame(render=True)
    
    # 最简单的网络
    class TinyActor(nn.Module):
        def __init__(self, input_dim=14, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, output_dim)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return torch.softmax(x, dim=-1)
    
    # 加载模型
    actor = TinyActor()
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"📦 检查点结构: {checkpoint.keys() if isinstance(checkpoint, dict) else '不是字典'}")
        
        # 尝试加载actor
        if isinstance(checkpoint, dict):
            if 'actor' in checkpoint:
                actor.load_state_dict(checkpoint['actor'])
            elif 'actor_state_dict' in checkpoint:
                actor.load_state_dict(checkpoint['actor_state_dict'])
            elif 'model' in checkpoint:
                actor.load_state_dict(checkpoint['model'])
            else:
                # 尝试所有键
                for key in checkpoint:
                    if isinstance(checkpoint[key], dict) and 'weight' in checkpoint[key]:
                        try:
                            actor.load_state_dict(checkpoint[key])
                            print(f"✅ 使用键 '{key}' 加载")
                            break
                        except:
                            continue
        else:
            actor.load_state_dict(checkpoint)
    except Exception as e:
        print(f"⚠️  加载模型时出错: {e}")
        print("使用随机初始化的模型")
    
    actor.eval()
    
    # 动作映射
    def get_action(state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = actor(state_tensor)
            action_idx = torch.argmax(probs).item()
            
            if action_idx == 0:
                # 移动
                return random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])
            else:
                # 瞄准
                return ACTION_GUN_LEFT if random.random() < 0.7 else ACTION_GUN_RIGHT
    
    # 测试循环
    state = game.reset()
    step = 0
    kills = 0
    
    print(f"\n🎮 开始测试")
    print("按 Q 退出")
    
    while step < 350 and not game.game_over:
        step += 1
        
        action = get_action(state)
        game.do_action(action)
        game.player.auto_shoot = True
        game.step()
        
        state = game.get_state()
        kills = game.score // 70 if game.score > 0 else 0
        
        # 显示当前状态
        print(f"步数: {step:3d} | 击杀: {kills:2d} | 得分: {game.score:4d}", end='\r')
        
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                print(f"\n🛑 手动退出")
                return
    
    pygame.quit()
    print(f"\n✅ 测试完成")
    print(f"最终击杀: {kills}")
    print(f"最终得分: {game.score}")

# 运行
if __name__ == "__main__":
    model_path = "./tank_ai_models_simple/ppo_gan_simple_ep300.pth"
    super_simple_test(model_path)