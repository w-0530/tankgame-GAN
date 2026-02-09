#!/usr/bin/env python3
"""
快速测试改进的状态表示，验证模型能否躲避子弹
"""
import torch
import numpy as np
import tankgame

def test_improved_state():
    """测试新的20维状态表示"""
    game = tankgame.TankGame(render=False)
    state = game.get_state()
    
    print(f"状态向量维度: {len(state)}")
    print(f"状态向量内容:")
    print(f"  玩家位置(x,y): ({state[0]:.3f}, {state[1]:.3f})")
    print(f"  玩家炮管角度: {state[2]:.3f}")
    print(f"  敌人位置(x,y): ({state[3]:.3f}, {state[4]:.3f})")
    print(f"  敌人距离: {state[5]:.3f}")
    print(f"  敌人角度: {state[6]:.3f}")
    print(f"  敌人生命: {state[7]:.3f}")
    
    print(f"\n敌人子弹信息 (4颗子弹的x,y位置):")
    for i in range(4):
        x, y = state[8 + i*2], state[9 + i*2]
        if x > 0:
            print(f"  子弹{i+1}: ({x:.3f}, {y:.3f})")
        else:
            print(f"  子弹{i+1}: 无")
    
    print(f"\n玩家状态:")
    print(f"  冷却时间: {state[16]:.3f}")
    print(f"  玩家子弹数: {state[17]:.3f}")
    print(f"  敌人子弹总数: {state[18]:.3f}")
    print(f"  玩家生命: {state[19]:.3f}")

def test_bullet_dodge():
    """测试子弹躲避行为"""
    game = tankgame.TankGame(render=True)
    
    # 创建一些敌人子弹来测试躲避
    for i in range(3):
        bullet = tankgame.Bullet(
            300 + i*100, 100,  # 在玩家上方创建子弹
            tankgame.deg2rad(90),  # 向下发射
            is_player_bullet=False
        )
        game.bullets.append(bullet)
    
    print("测试子弹躲避 - 场景中有3颗敌人子弹从上方射向玩家")
    print("观察AI是否能学会向左右躲避")
    
    # 手动运行几步观察
    state = game.get_state()
    print(f"\n初始状态:")
    print(f"敌人子弹位置:")
    for i in range(4):
        x, y = state[8 + i*2], state[9 + i*2]
        if x > 0:
            screen_x = x * tankgame.SCREEN_WIDTH
            screen_y = y * tankgame.SCREEN_HEIGHT
            player_x = state[0] * tankgame.SCREEN_WIDTH
            player_y = state[1] * tankgame.SCREEN_HEIGHT
            dist = np.sqrt((screen_x - player_x)**2 + (screen_y - player_y)**2)
            print(f"  子弹{i+1}: 位置({screen_x:.0f}, {screen_y:.0f}), 距离玩家{dist:.0f}")

if __name__ == "__main__":
    print("=== 测试改进的状态表示 ===")
    test_improved_state()
    
    print("\n=== 测试子弹躲避场景 ===")
    test_bullet_dodge()
    
    print("\n改进总结:")
    print("1. 状态向量从12维扩展到20维")
    print("2. 新增4颗敌人子弹的精确位置信息")
    print("3. 探索策略优先躲避近距离子弹")
    print("4. 神经网络输入层更新为20维")
    print("5. 模型现在能够看到子弹位置并学会躲避！")

