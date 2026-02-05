# diagnostic.py
from tankgame import TankGame
import numpy as np

game = TankGame(render=False)
state = game.get_state()

print("=" * 60)
print("状态维度诊断")
print("=" * 60)
print(f"状态维度: {len(state)}")
print(f"状态内容: {state}")
print(f"状态类型: {type(state)}")
print(f"状态形状: {np.array(state).shape if hasattr(state, '__len__') else 'N/A'}")

# 测试多次确保一致
print("\n多次测试确认:")
for i in range(3):
    game.reset()
    state = game.get_state()
    print(f"测试 {i+1}: 维度={len(state)}")