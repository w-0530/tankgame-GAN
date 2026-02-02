import os
import pygame
import torch
from tankgame import TankGame
from tank_gan_ppo import PPO_GAN, STATE_DIM, ACTION_DIM, MAX_STEP

RENDER = True
MODEL_PATH = "./tank_ai_models/ppo_gan_fire_ep1000.pth"

def test_model(model_path):
    assert os.path.exists(model_path), f"模型不存在: {model_path}"

    pygame.init()
    game = TankGame(render=RENDER)

    agent = PPO_GAN(STATE_DIM, ACTION_DIM)
    agent.load(model_path)

    agent.actor.eval()          # 评估模式
    agent.current_epoch = 10**9 # 强制关闭 GAN 奖励（如果内部有判断）

    state = game.reset()
    agent.reset_combat_state()

    clock = pygame.time.Clock()
    step = 0
    done = False

    print(f"\n🎮 测试模型：{model_path}")

    while not done:
        step += 1
        clock.tick(60)

        # ⭐ 只用策略，不用随机、不训练
        action = agent.actor.get_best_action(state)

        game.do_action(action)
        _, game_done = game.step()

        state = game.get_state()
        done = game_done or step >= MAX_STEP

        if RENDER:
            game.render()
            pygame.display.update()

    pygame.quit()
    print(f"✅ 测试结束 | 步数={step}")
    print(f"💀 存活={game.player.alive} | 击杀={game.score // 70}")

if __name__ == "__main__":
    test_model(MODEL_PATH)
