import os
import pygame
import torch

from tankgame import TankGame
from tank_gan_ppo import (
    PPO_GAN_Simple,
    STATE_DIM,
    MAX_STEP
)

# ===================== 配置 =====================
RENDER = True
FPS = 60
MODEL_PATH = "./tank_ai_models_simple/ppo_gan_simple_ep1200.pth"
# ===============================================


def test_model(model_path):
    assert os.path.exists(model_path), f"❌ 模型不存在: {model_path}"

    pygame.init()
    game = TankGame(render=RENDER)
    clock = pygame.time.Clock()

    # ---------- agent ----------
    agent = PPO_GAN_Simple()
    agent.load(model_path)

    # ⭐ 强制评估模式（保险）
    agent.actor.eval()
    agent.critic.eval()

    # ---------- reset ----------
    state = game.reset()
    agent.reset_combat_state()

    step = 0
    total_reward = 0.0
    kill_num = 0

    print(f"\n🎮 开始测试模型：{model_path}")
    print("💡 按 Q 或关闭窗口退出\n")

    # ===================== 主循环 =====================
    while step < MAX_STEP and not game.game_over:
        step += 1
        clock.tick(FPS)

        # ✅ 测试阶段：只用最优策略 → 游戏动作
        with torch.no_grad():
            action = agent.actor.get_best_action(state)

        game.do_action(action)
        game.player.auto_shoot = True
        game.step()

        state = game.get_state()
        kill_num = game.score // 70 if game.score > 0 else 0

        if RENDER:
            game.render()
            pygame.display.flip()

        # ---------- 事件 ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                pygame.quit()
                print("\n🛑 测试手动退出")
                return

    pygame.quit()

    # ===================== 结果 =====================
    print("\n✅ 测试结束")
    print(f"📊 步数：{step}")
    print(f"🏆 击杀数：{kill_num}")
    print(f"💀 存活：{game.player.alive}")
    print(f"🎯 得分：{game.score}")
    print(f"📈 结果：{'胜利' if kill_num >= 8 else '失败'}（胜利条件：击杀≥8）")


if __name__ == "__main__":
    test_model(MODEL_PATH)
