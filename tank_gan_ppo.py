import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pygame
import math
from tqdm import tqdm
from collections import deque
from tankgame import TankGame

# ====================== 基础设置 ======================
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

DEVICE = torch.device("cpu")

STATE_DIM = 14          # ⚠️ 会在运行期校验
ACTION_DIM = 8          # 0~7

# ====================== PPO 超参数 ======================
PPO_LR = 3e-4
PPO_LR_DECAY_EPOCH = 200
PPO_LR_DECAY_RATE = 0.5
GAMMA = 0.98
LAMBDA = 0.85
EPS_CLIP = 0.25
BATCH_SIZE = 32
ENT_COEF = 0.15
ADV_CLIP = 1.0
VALUE_CLIP = 1.0

# ====================== 奖励参数 ======================
HIT_REWARD_MIN = 5.0
HIT_REWARD_MAX = 6.0
KILL_REWARD_MIN = 70.0
KILL_REWARD_MAX = 70.0
CLOSE_NO_SHOOT_PUNISH = 0.99
DEATH_PUNISH = 50.0
IDLE_PUNISH = 0.15
MOVE_TOWARD_ENEMY_REW = 0.5
SINGLE_ACTION_PUNISH = 0.7
AIM_ACTION_REW = 0.2

CLOSE_DIST_THRESHOLD = 220
DIST_JUDGE_THRESHOLD = 2

MAX_STEP = 350
NO_COMBAT_STEP = 45

# ====================== GAN ======================
GAN_LR = 4e-5
GAN_TRAIN_TIMES = 1
GAN_BONUS_WEIGHT = 0.05
GAN_STOP_EPOCH = 30

MEMORY_CAPACITY = 60000
DEMO_MEMORY_CAPACITY = 4000
TRAIN_EPISODES = 100
SAVE_INTERVAL = 20

RENDER_TRAIN = False
RENDER_TEST = True
TRAIN_STEP_INTERVAL = 3

os.makedirs("./tank_ai_models", exist_ok=True)

# =========================================================
# PPO 网络
# =========================================================
class PPO_Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

    def get_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self(s)
        a = torch.multinomial(probs, 1).item()
        return a, probs[0, a].item()

    def get_best_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self(s)
        return torch.argmax(probs, dim=1).item()


class PPO_Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# GAN 判别器
# =========================================================
class GAN_Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

# =========================================================
# PPO + GAN Agent
# =========================================================
class PPO_GAN:
    def __init__(self, state_dim, action_dim):
        self.actor = PPO_Actor(state_dim, action_dim)
        self.critic = PPO_Critic(state_dim)
        self.gan_d = GAN_Discriminator(state_dim, action_dim)

        self.ppo_opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=PPO_LR
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.ppo_opt, PPO_LR_DECAY_EPOCH, PPO_LR_DECAY_RATE
        )

        self.gan_opt = optim.Adam(self.gan_d.parameters(), lr=GAN_LR)
        self.gan_loss_fn = nn.BCELoss()

        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.demo_memory = deque(maxlen=DEMO_MEMORY_CAPACITY)

        self.current_epoch = 0
        self.reset_combat_state()

    def reset_combat_state(self):
        self.no_combat_step = 0
        self.hit_count = 0
        self.kill_flag = False
        self.death_flag = False
        self.last_enemy_dist = float("inf")
        self.last_score = 0
        self.enemy_lives_cache = {}
        self.last_action = -1

    def one_hot(self, actions):
        oh = torch.zeros(len(actions), ACTION_DIM)
        oh[range(len(actions)), actions] = 1.0
        return oh

    def store_ppo(self, s, a, p, r, ns, d, is_combat):
        self.memory.append((s, a, p, r, ns, d))
        self.no_combat_step = 0 if is_combat else self.no_combat_step + 1

    def compute_gae(self, r, d, v, nv):
        gae = 0
        adv = torch.zeros_like(r)
        for i in reversed(range(len(r))):
            delta = r[i] + GAMMA * nv[i] * (1 - d[i]) - v[i]
            gae = delta + GAMMA * LAMBDA * (1 - d[i]) * gae
            adv[i] = gae
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        return torch.clamp(adv, -ADV_CLIP, ADV_CLIP)

    def train_ppo(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        batch = np.random.choice(len(self.memory), BATCH_SIZE, False)
        data = [self.memory[i] for i in batch]

        s = torch.FloatTensor([d[0] for d in data])
        a = torch.LongTensor([d[1] for d in data]).unsqueeze(1)
        old_p = torch.FloatTensor([d[2] for d in data]).unsqueeze(1)
        r = torch.FloatTensor([d[3] for d in data]).unsqueeze(1)
        ns = torch.FloatTensor([d[4] for d in data])
        d = torch.FloatTensor([d[5] for d in data]).unsqueeze(1)

        with torch.no_grad():
            v = torch.clamp(self.critic(s), -VALUE_CLIP, VALUE_CLIP)
            nv = torch.clamp(self.critic(ns), -VALUE_CLIP, VALUE_CLIP)

        adv = self.compute_gae(r, d, v, nv)
        target_v = adv + v

        probs = self.actor(s)
        new_p = probs.gather(1, a)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        ratio = torch.exp(torch.log(new_p + 1e-8) - torch.log(old_p + 1e-8))
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(self.critic(s), target_v)

        loss = actor_loss + 0.5 * critic_loss - ENT_COEF * entropy

        self.ppo_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), 1.2
        )
        self.ppo_opt.step()

        if self.current_epoch >= PPO_LR_DECAY_EPOCH:
            self.lr_scheduler.step()

        return loss.item()

    def save(self, ep):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "gan": self.gan_d.state_dict()
        }, f"./tank_ai_models/ppo_gan_fire_ep{ep}.pth")

# =========================================================
# 工具函数（关键修复版）
# =========================================================
def get_nearest_enemy(game):
    enemies = [e for e in game.enemies if e.alive]
    if not enemies:
        return None
    return min(enemies, key=lambda e: math.hypot(game.player.x - e.x, game.player.y - e.y))

def calculate_enemy_dist(game):
    e = get_nearest_enemy(game)
    if not e:
        return float("inf")
    return math.hypot(game.player.x - e.x, game.player.y - e.y)

def check_hit_kill(game, agent):
    is_hit = False
    is_kill = False
    score = game.score

    if score > agent.last_score:
        diff = score - agent.last_score
        if diff >= 20:
            is_kill = True
        else:
            is_hit = True

    enemy = get_nearest_enemy(game)
    if enemy:
        last = agent.enemy_lives_cache.get(enemy, enemy.lives)
        if enemy.lives < last:
            is_hit = True
            if enemy.lives <= 0:
                is_kill = True
        agent.enemy_lives_cache[enemy] = enemy.lives

    agent.last_score = score
    return is_hit, is_kill

# =========================================================
# 奖励函数（修复版）
# =========================================================
def combat_reward(game, action, step, agent):
    final_reward = 0.0
    is_combat = False

    shoot = 7
    idle = 0
    aim = [5, 6]
    move = [1, 2, 3, 4]

    is_hit, is_kill = check_hit_kill(game, agent)
    dist = calculate_enemy_dist(game)

    if agent.last_action == action and action in move + aim and step > 5:
        final_reward -= SINGLE_ACTION_PUNISH
        is_combat = True
    agent.last_action = action

    if action in aim:
        final_reward += AIM_ACTION_REW
        is_combat = True

    if is_kill and not agent.kill_flag:
        final_reward += np.random.uniform(KILL_REWARD_MIN, KILL_REWARD_MAX)
        agent.kill_flag = True
        is_combat = True

    if is_hit:
        final_reward += np.random.uniform(HIT_REWARD_MIN, HIT_REWARD_MAX)
        is_combat = True

    if not game.player.alive and not agent.death_flag:
        final_reward -= DEATH_PUNISH
        agent.death_flag = True

    if dist < CLOSE_DIST_THRESHOLD and action != shoot:
        final_reward -= CLOSE_NO_SHOOT_PUNISH
        is_combat = True

    if dist < agent.last_enemy_dist - DIST_JUDGE_THRESHOLD:
        final_reward += MOVE_TOWARD_ENEMY_REW

    agent.last_enemy_dist = dist

    if action == idle:
        final_reward -= IDLE_PUNISH

    return np.clip(final_reward, -80, 80), is_combat

# =========================================================
# 训练入口
# =========================================================
def train_ai():
    pygame.init()
    game = TankGame(render=RENDER_TRAIN)
    agent = PPO_GAN(STATE_DIM, ACTION_DIM)

    state = game.reset()
    assert len(state) == STATE_DIM, f"STATE_DIM={STATE_DIM}, get_state()={len(state)}"
    print(f"✅ STATE_DIM 校验通过：{STATE_DIM}")

    clock = pygame.time.Clock()
    for ep in tqdm(range(1, TRAIN_EPISODES + 1)):
        agent.current_epoch = ep
        agent.reset_combat_state()
        state = game.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            step += 1
            clock.tick(120)

            if step >= MAX_STEP or agent.no_combat_step >= NO_COMBAT_STEP:
                break

            action, prob = agent.actor.get_action(state)
            game.do_action(action)
            _, game_done = game.step()

            r, is_combat = combat_reward(game, action, step, agent)
            next_state = game.get_state()

            agent.store_ppo(state, action, prob, r, next_state, game_done, is_combat)
            total_reward += r

            if step % TRAIN_STEP_INTERVAL == 0:
                agent.train_ppo()

            state = next_state
            done = game_done

        if ep % SAVE_INTERVAL == 0:
            agent.save(ep)

    agent.save(TRAIN_EPISODES)
    pygame.quit()

# =========================================================
if __name__ == "__main__":
    train_ai()
