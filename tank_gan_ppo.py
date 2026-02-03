import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pygame
import math
import random
from tqdm import tqdm
from collections import deque

# 导入游戏核心类+常量+动作（修复WALL_SIZE未定义）
from tankgame import (
    TankGame, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_GUN_LEFT, ACTION_GUN_RIGHT, WALL_SIZE
)

# ====================== 基础配置 =======================
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 训练设备：{DEVICE} | CUDA可用：{torch.cuda.is_available()}")

# 核心维度（AI仅输出0/1动作类型，映射游戏实际动作）
STATE_DIM = 14
ACTION_DIM = 2  # 0=移动 1=瞄准
DEMO_VEC_DIM = STATE_DIM + ACTION_DIM

# ====================== 超参数（核心修复：强化瞄准、弱化移动） ======================
# PPO超参数
PPO_LR = 3e-4
GAMMA = 0.98
LAMBDA = 0.85
EPS_CLIP = 0.25
BATCH_SIZE = 32
ENT_COEF = 0.05  # 降低探索系数，避免无意义乱动
MAX_STEP = 350
# GAN超参数（保守权重，不主导训练）
GAN_LR = 4e-5
GAN_UPDATE_INTERVAL = 5
GAN_REWARD_WEIGHT = 0.05  # 进一步降低GAN权重

# ====================== 奖励/惩罚配置（核心修复：瞄准零门槛+梯度化 | 向敌奖励） ======================
REWARD_DIRECT_SHOT_POS = 1.5     # 降低移动奖励，避免移动主导
REWARD_DODGE_BULLET = 0.8        # 降低躲子弹奖励，消极防御不优先
REWARD_AIM_GRADIENT = 3.0        # 零门槛梯度瞄准奖励（只要转炮管就给）
REWARD_AIM_GOOD = 4.0            # 较好瞄准奖励（误差<0.3，易触发）
REWARD_AIM_PERFECT = 6.0         # 完美瞄准奖励（门槛降低到0.2）
REWARD_TOWARD_ENEMY = 1.2        # 核心：向敌人移动奖励，引导有意识跑位
PUNISH_NO_KILL = 0.8             # 降低未击杀惩罚，给AI瞄准时间
PUNISH_IDLE = 0.3                # 大幅降低重复动作惩罚，允许连续瞄准
PUNISH_BEEN_HIT = 20.0           # 降低被击中惩罚，训练初期不怕失误
# 判定阈值（全量降低门槛，训练初期易触发）
BULLET_DODGE_DIST = 60           # 提高躲子弹安全距离
NO_KILL_STEP_THRESH = 30         # 提高未击杀惩罚步数
CLOSE_DIST_THRESH = 250          # 扩大敌人暴露距离
RAYCAST_STEP = 15                # 降低射线检测精度，直射位易判定
AIM_GOOD_THRESH = 0.3            # 较好瞄准阈值（30%误差）
AIM_PERFECT_THRESH = 0.2         # 完美瞄准阈值（20%误差，易触发）

# ====================== 训练配置（快速验证，100轮见效果） ======================
MEMORY_CAPACITY = 40000  # 减小经验池，训练更新更快
DEMO_MEMORY_CAPACITY = 3000
TRAIN_EPISODES = 1200     # 减少训练轮数，快速验证
SAVE_INTERVAL = 50       # 提高保存频率，每50轮测试一次
RENDER_TRAIN = False
RENDER_TEST = True

# 创建保存目录
os.makedirs("./tank_ai_models_simple", exist_ok=True)
os.makedirs("./tank_demo_data_simple", exist_ok=True)

# =========================================================
# PPO 经验缓冲区
# =========================================================
class PPOMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def add(self, s, a, r, ns, d, p):
        self.memory.append((s, a, r, ns, d, p))
    
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        data = [self.memory[i] for i in idx]
        s = torch.FloatTensor([d[0] for d in data]).to(DEVICE)
        a = torch.LongTensor([d[1] for d in data]).unsqueeze(1).to(DEVICE)
        r = torch.FloatTensor([d[2] for d in data]).unsqueeze(1).to(DEVICE)
        ns = torch.FloatTensor([d[3] for d in data]).to(DEVICE)
        d = torch.FloatTensor([d[4] for d in data]).unsqueeze(1).to(DEVICE)
        p = torch.FloatTensor([d[5] for d in data]).unsqueeze(1).to(DEVICE)
        return s, a, r, ns, d, p
    
    def __len__(self):
        return len(self.memory)

# =========================================================
# 专家经验缓冲区
# =========================================================
class DemoMemory:
    def __init__(self, capacity, vec_dim):
        self.capacity = capacity
        self.vec_dim = vec_dim
        self.memory = deque(maxlen=capacity)
    
    def add(self, demo_vec):
        assert demo_vec.shape[0] == self.vec_dim, f"维度错误：实际{demo_vec.shape[0]}，期望{self.vec_dim}"
        self.memory.append(demo_vec)
    
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        demo_array = np.array([self.memory[i] for i in idx], dtype=np.float32)
        return torch.FloatTensor(demo_array).to(DEVICE)
    
    def save_to_npy(self, npy_path):
        np.save(npy_path, np.array(self.memory, dtype=np.float32))
        print(f"💾 专家经验保存至：{npy_path}")
    
    def load_from_npy(self, npy_path):
        if os.path.exists(npy_path):
            demo_array = np.load(npy_path)
            for vec in demo_array:
                self.add(vec)
            print(f"📚 加载专家经验：{len(demo_array)}条 | 维度：{demo_array.shape[1]}")
    
    def __len__(self):
        return len(self.memory)

# =========================================================
# PPO 网络（核心修改：LeakyReLU→ReLU，适配浅层模型+初始化优化）
# =========================================================
class PPO_Actor(nn.Module):
    def __init__(self, state_dim, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),  # 替换LeakyReLU，梯度更直接
            nn.Linear(64, 32),
            nn.ReLU(),  # 替换LeakyReLU
            nn.Linear(32, action_dim)
        ).to(DEVICE)
        self.MOVE_ACTION_LIST = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
        self.AIM_ACTION_LIST = [ACTION_GUN_LEFT, ACTION_GUN_RIGHT]
        # 核心：瞄准动作探索偏向，强制AI探索瞄准（打破移动死循环）
        self.aim_explore_bias = 0.4
        # 浅层网络初始化优化：偏置置0，避免初始值干扰
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        logits = self.net(x.to(DEVICE))
        # 给瞄准动作（索引1）加偏向，提升其被选中的概率
        logits[:, 1] += self.aim_explore_bias
        return F.softmax(logits, dim=-1)

    def get_action(self, state):
        """训练用：返回游戏动作、AI0/1动作、动作概率 | 训练期也加瞄准方向偏好"""
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self(s)
        ai_action = torch.multinomial(probs, 1).item()
        # 训练期瞄准方向6:4偏好，让炮管连续转动，而非随机乱转
        if ai_action == 0:
            game_action = random.choice(self.MOVE_ACTION_LIST)
        else:
            game_action = ACTION_GUN_LEFT if random.random() < 0.6 else ACTION_GUN_RIGHT
        return game_action, ai_action, probs[0, ai_action].item()

    def get_best_action(self, state):
        """测试用：瞄准动作7:3方向偏好，炮管连续转不抖动"""
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self(s)
        ai_action = torch.argmax(probs, dim=1).item()
        if ai_action == 0:
            return random.choice(self.MOVE_ACTION_LIST)
        else:
            return ACTION_GUN_LEFT if random.random() < 0.7 else ACTION_GUN_RIGHT

class PPO_Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),  # 替换LeakyReLU
            nn.Linear(64, 32),
            nn.ReLU(),  # 替换LeakyReLU
            nn.Linear(32, 1)
        ).to(DEVICE)
        # 浅层网络初始化优化：偏置置0
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x.to(DEVICE))

# =========================================================
# GAN 判别器（核心修改：LeakyReLU→ReLU）
# =========================================================
class GAILGAN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),  # 替换LeakyReLU
            nn.Linear(128, 1)
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=GAN_LR)
        # 初始化优化：偏置置0
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.discriminator(x.to(DEVICE))
    
    def train_step(self, agent_batch, expert_batch):
        expert_logits = self(expert_batch)
        agent_logits = self(agent_batch)
        loss_expert = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits).to(DEVICE))
        loss_agent = F.binary_cross_entropy_with_logits(agent_logits, torch.zeros_like(agent_logits).to(DEVICE))
        loss = 0.5 * (loss_expert + loss_agent)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            gan_reward = -torch.log(torch.sigmoid(agent_logits) + 1e-8).mean().item()
        return loss.item(), gan_reward

# =========================================================
# PPO-GAN 智能体
# =========================================================
class PPO_GAN_Simple:
    def __init__(self):
        self.actor = PPO_Actor(STATE_DIM, ACTION_DIM)
        self.critic = PPO_Critic(STATE_DIM)
        self.ppo_opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=PPO_LR)
        self.ppo_memory = PPOMemory(MEMORY_CAPACITY)
        self.demo_memory = DemoMemory(DEMO_MEMORY_CAPACITY, DEMO_VEC_DIM)
        self.gan = GAILGAN(DEMO_VEC_DIM)
        self.reset_combat_state()

    def reset_combat_state(self):
        self.no_kill_step = 0
        self.last_action = -1
        self.last_player_pos = (0, 0)

    def one_hot(self, actions):
        oh = torch.zeros(len(actions), ACTION_DIM).to(DEVICE)
        oh[range(len(actions)), actions] = 1.0
        return oh
    
    def compute_gae(self, r, d, v, nv):
        gae = 0
        adv = torch.zeros_like(r).to(DEVICE)
        for i in reversed(range(len(r))):
            delta = r[i] + GAMMA * nv[i] * (1 - d[i]) - v[i]
            gae = delta + GAMMA * LAMBDA * (1 - d[i]) * gae
            adv[i] = gae
        return adv.clamp(-1.0, 1.0)

    def train_ppo(self):
        """PPO训练：索引已匹配（a是0/1），无越界"""
        batch = self.ppo_memory.sample(BATCH_SIZE)
        if batch is None:
            return 0.0
        s, a, r, ns, d, old_p = batch
        with torch.no_grad():
            v = self.critic(s)
            nv = self.critic(ns)
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
        total_loss = actor_loss + 0.5 * critic_loss - ENT_COEF * entropy
        self.ppo_opt.zero_grad()
        total_loss.backward()
        self.ppo_opt.step()
        return total_loss.item()

    def save(self, ep):
        save_path = f"./tank_ai_models_simple/ppo_gan_simple_ep{ep}.pth"
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, save_path)
        print(f"\n💾 模型保存：{save_path}")

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        print(f"✅ 加载模型：{model_path}")

# =========================================================
# 工具函数（新增向敌移动判定）
# =========================================================
def get_nearest_enemy(game):
    enemies_alive = [e for e in game.enemies if e.alive]
    if not enemies_alive:
        return None
    distances = [math.hypot(e.x - game.player.x, e.y - game.player.y) for e in enemies_alive]
    return enemies_alive[np.argmin(distances)]

def raycast_obstacle(game, start_pos, end_pos):
    sx, sy = start_pos
    ex, ey = end_pos
    dx = ex - sx
    dy = ey - sy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return True
    step_x = (dx / dist) * RAYCAST_STEP
    step_y = (dy / dist) * RAYCAST_STEP
    current_x, current_y = sx, sy
    for _ in range(int(dist // RAYCAST_STEP) + 1):
        current_x += step_x
        current_y += step_y
        for wall in game.walls:
            wall_rect = pygame.Rect(wall[0], wall[1], WALL_SIZE, WALL_SIZE)
            if wall_rect.collidepoint(current_x, current_y):
                return True
    return False

def is_in_direct_shot_position(game):
    enemy = get_nearest_enemy(game)
    if not enemy or not game.player.alive:
        return False
    return not raycast_obstacle(game, (game.player.x, game.player.y), (enemy.x, enemy.y))

def is_dodging_bullet(game):
    if not hasattr(game, "bullets") or len(game.bullets) == 0:
        return True
    player_x, player_y = game.player.x, game.player.y
    for bullet in game.bullets:
        if not bullet.is_player_bullet and math.hypot(bullet.x - player_x, bullet.y - player_y) <= BULLET_DODGE_DIST:
            return False
    return True

def is_enemy_exposed(game):
    enemy = get_nearest_enemy(game)
    if not enemy:
        return False
    return math.hypot(enemy.x - game.player.x, enemy.y - game.player.y) < CLOSE_DIST_THRESH

def calculate_aim_error(game):
    enemy = get_nearest_enemy(game)
    if not enemy:
        return 1.0
    dx = enemy.x - game.player.x
    dy = enemy.y - game.player.y
    target_angle = math.atan2(-dy, dx) % (2 * math.pi)
    current_angle = game.player.aim_angle % (2 * math.pi)
    angle_error = abs(current_angle - target_angle)
    angle_error = min(angle_error, 2 * math.pi - angle_error)
    return angle_error / math.pi  # 归一化到0~1

# 核心新增：判定是否向敌人移动
def is_toward_enemy(game, last_player_pos):
    enemy = get_nearest_enemy(game)
    if not enemy or not game.player.alive:
        return False
    last_x, last_y = last_player_pos
    dist_last = math.hypot(enemy.x - last_x, enemy.y - last_y)
    dist_current = math.hypot(enemy.x - game.player.x, enemy.y - game.player.y)
    return (dist_last - dist_current) > 1e-3  # 距离减少=向敌移动

# =========================================================
# 奖励函数（完全重写：零门槛梯度瞄准+向敌移动引导）
# =========================================================
def get_env_reward(game, action, agent):
    final_reward = 0.0
    enemy = get_nearest_enemy(game)
    enemy_visible = enemy is not None and game.player.alive
    current_player_pos = (game.player.x, game.player.y)

    # 1. 瞄准动作奖励（核心：零门槛+梯度化，只要有敌人就给奖励）
    if action in [ACTION_GUN_LEFT, ACTION_GUN_RIGHT] and enemy_visible:
        aim_error = calculate_aim_error(game)
        # 零门槛梯度奖励：误差越小，奖励越高
        final_reward += REWARD_AIM_GRADIENT * (1 - aim_error)
        # 较好瞄准奖励（误差<0.3，易触发）
        if aim_error < AIM_GOOD_THRESH:
            final_reward += REWARD_AIM_GOOD
        # 完美瞄准奖励（误差<0.2，叠加高额奖励）
        if aim_error < AIM_PERFECT_THRESH:
            final_reward += REWARD_AIM_PERFECT

    # 2. 移动动作奖励（引导有意识跑位：优先向敌移动）
    if action in [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT] and enemy_visible:
        # 核心：向敌人移动就给奖励，引导AI主动靠近
        if is_toward_enemy(game, agent.last_player_pos):
            final_reward += REWARD_TOWARD_ENEMY
        # 直射位奖励（次要）
        if is_in_direct_shot_position(game):
            final_reward += REWARD_DIRECT_SHOT_POS
        # 躲子弹奖励（最次要）
        if is_dodging_bullet(game):
            final_reward += REWARD_DODGE_BULLET

    # 3. 惩罚项（大幅弱化，不压制探索）
    if enemy_visible:
        agent.no_kill_step += 1
        if agent.no_kill_step >= NO_KILL_STEP_THRESH:
            final_reward -= PUNISH_NO_KILL
            agent.no_kill_step = NO_KILL_STEP_THRESH - 10
    # 重复动作惩罚（允许连续瞄准/向敌移动）
    if agent.last_action == action and agent.last_action != -1:
        final_reward -= PUNISH_IDLE
    # 被击中惩罚（降低，不怕失误）
    if hasattr(game.player, 'been_hit') and game.player.been_hit:
        final_reward -= PUNISH_BEEN_HIT
        game.player.been_hit = False

    # 4. 自动开火绑定瞄准精度：只有较好瞄准才开火（避免无脑射击）
    if enemy_visible:
        aim_error = calculate_aim_error(game)
        game.player.auto_shoot = aim_error < AIM_GOOD_THRESH
    else:
        game.player.auto_shoot = False

    # 更新代理状态
    agent.last_action = action
    agent.last_player_pos = current_player_pos
    return np.clip(final_reward, -10, 20)

# =========================================================
# 生成专家经验
# =========================================================
def generate_demo_data(demo_memory, demo_num=300):  # 减少采集数量，快速完成
    print("\n🎮 生成专家经验 | 按键：WASD=移动 | ←→=瞄准 | ESC退出")
    print(f"🎯 目标采集：{demo_num}条有效经验（重点多转炮管瞄准！）")
    game = TankGame(render=True)
    state = game.reset()
    clock = pygame.time.Clock()
    running = True

    while running and len(demo_memory) < demo_num:
        clock.tick(60)
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                break
        if not running:
            break
        # 手动操作映射AI动作类型
        keys = pygame.key.get_pressed()
        ai_action_type = 0
        if keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d]:
            ai_action_type = 0
            action = ACTION_UP if keys[pygame.K_w] else ACTION_DOWN if keys[pygame.K_s] else ACTION_LEFT if keys[pygame.K_a] else ACTION_RIGHT
        elif keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
            ai_action_type = 1
            action = ACTION_GUN_LEFT if keys[pygame.K_LEFT] else ACTION_GUN_RIGHT
        # 采集经验（放宽条件，只要存活就采集，重点是瞄准动作）
        if game.player.alive:
            state_np = np.asarray(state, dtype=np.float32).flatten()
            action_one_hot = np.eye(ACTION_DIM, dtype=np.float32)[ai_action_type]
            demo_vec = np.concatenate([state_np, action_one_hot])
            demo_memory.add(demo_vec)
            if len(demo_memory) % 50 == 0:
                print(f"📈 已采集：{len(demo_memory)}/{demo_num} 条")
        # 执行动作
        game.do_action(action)
        game.player.auto_shoot = True
        game.step()
        state = game.get_state()
        if game.game_over:
            state = game.reset()
            print(f"🔄 游戏重置 | 继续采集经验...")
    # 保存经验
    demo_memory.save_to_npy("./tank_demo_data_simple/demo_memory_simple.npy")
    pygame.quit()
    print(f"\n✅ 专家经验生成完成！实际采集：{len(demo_memory)}条")

# =========================================================
# 训练入口
# =========================================================
def train_ai(load_model_path=None):
    pygame.init()
    game = TankGame(render=RENDER_TRAIN)
    agent = PPO_GAN_Simple()

    # 加载/采集专家经验
    demo_path = "./tank_demo_data_simple/demo_memory_simple.npy"
    if os.path.exists(demo_path):
        agent.demo_memory.load_from_npy(demo_path)
    else:
        print(f"⚠️  未找到专家经验，开始手动采集...")
        generate_demo_data(agent.demo_memory, 300)  # 只采集300条，快速完成
    # 加载预训练模型
    if load_model_path and os.path.exists(load_model_path):
        agent.load(load_model_path)
    # 维度校验
    assert len(game.get_state()) == STATE_DIM, f"状态维度不匹配！游戏{len(game.get_state())} vs 配置{STATE_DIM}"
    print(f"\n🚀 正式开始训练 | 总轮数：{TRAIN_EPISODES} | GAN奖励权重：{GAN_REWARD_WEIGHT}")
    print(f"📌 核心奖励：零门槛瞄准+{REWARD_AIM_GRADIENT} | 向敌移动+{REWARD_TOWARD_ENEMY}")
    pbar = tqdm(range(1, TRAIN_EPISODES + 1), desc="训练进度")

    for ep in pbar:
        state = game.reset()
        agent.reset_combat_state()
        total_reward = 0.0
        step = 0
        ppo_loss_sum = 0.0
        ppo_train_count = 0
        gan_reward_sum = 0.0

        while step < MAX_STEP and not game.game_over:
            step += 1
            # 获取动作（训练期带瞄准方向偏好）
            game_action, ai_action, action_prob = agent.actor.get_action(state)
            game.do_action(game_action)
            _, done = game.step()
            # 计算核心环境奖励（零门槛瞄准+向敌引导）
            base_reward = get_env_reward(game, game_action, agent)
            next_state = game.get_state()

            # GAN奖励融合（小权重，辅助训练）
            gan_reward = 0.0
            if step % GAN_UPDATE_INTERVAL == 0 and len(agent.ppo_memory) >= BATCH_SIZE:
                batch = agent.ppo_memory.sample(BATCH_SIZE)
                if batch is not None:
                    s_ppo, a_ppo, _, _, _, _ = batch
                    ai_action_onehot = agent.one_hot(a_ppo.squeeze(1))
                    agent_batch = torch.cat([s_ppo, ai_action_onehot], dim=-1)
                    expert_batch = agent.demo_memory.sample(BATCH_SIZE)
                    gan_loss, gan_reward = agent.gan.train_step(agent_batch, expert_batch)
                    gan_reward_sum += gan_reward
            # 总奖励 = 环境奖励 + 小权重GAN奖励
            total_reward_step = base_reward + GAN_REWARD_WEIGHT * gan_reward

            # 存储经验
            agent.ppo_memory.add(state, ai_action, total_reward_step, next_state, done, action_prob)
            total_reward += total_reward_step

            # 训练PPO
            if step % 3 == 0:
                ppo_loss = agent.train_ppo()
                ppo_loss_sum += ppo_loss
                ppo_train_count += 1
            
            # 更新状态
            state = next_state

        # 进度条展示
        avg_ppo_loss = ppo_loss_sum / max(ppo_train_count, 1)
        avg_gan_reward = gan_reward_sum / max(step // GAN_UPDATE_INTERVAL, 1)
        pbar.set_postfix({
            "总奖励": f"{total_reward:.2f}",
            "PPO损失": f"{avg_ppo_loss:.4f}",
            "平均GAN奖励": f"{avg_gan_reward:.3f}",
            "经验池": f"{len(agent.ppo_memory)}/{MEMORY_CAPACITY}"
        })
        # 保存模型
        if ep % SAVE_INTERVAL == 0:
            agent.save(ep)

    # 训练完成
    agent.save(TRAIN_EPISODES)
    pygame.quit()
    print(f"\n🎉 训练全部完成！最终模型保存至：./tank_ai_models_simple/ppo_gan_simple_ep{TRAIN_EPISODES}.pth")

# =========================================================
# 测试入口
# =========================================================
def test_ai(model_path):
    pygame.init()
    game = TankGame(render=RENDER_TEST)
    agent = PPO_GAN_Simple()
    agent.load(model_path)

    state = game.reset()
    agent.reset_combat_state()
    clock = pygame.time.Clock()
    total_reward = 0.0
    step = 0
    kill_num = 0
    print(f"\n🎮 AI测试启动 | 模型：{model_path} | 最大步数：{MAX_STEP}")
    print(f"💡 按 Q 或 关闭窗口 退出测试 | 瞄准偏好：70%左 | 30%右")

    while step < MAX_STEP and not game.game_over:
        step += 1
        clock.tick(60)
        # 调用带方向偏好的最优动作
        action = agent.actor.get_best_action(state)
        game.do_action(action)
        game.step()
        reward = get_env_reward(game, action, agent)
        total_reward += reward
        state = game.get_state()
        kill_num = game.score // 70 if game.score > 0 else 0

        # 退出监听
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                print(f"\n🛑 测试手动退出")
                return

    # 测试结果
    pygame.quit()
    print(f"\n✅ 测试结束 | 最终统计：")
    print(f"📊 步数：{step} | 总奖励：{total_reward:.2f} | 总得分：{game.score}")
    print(f"🏆 击杀数：{kill_num} | 剩余生命：{game.player.lives}")
    print(f"📈 测试结果：{'胜利' if kill_num >=8 else '失败'}（胜利条件：击杀≥8）")

# =========================================================
# 主函数（训练/测试一键切换）
# =========================================================
if __name__ == "__main__":
    TRAIN_MODE = True  # True=训练，False=测试
    PRETRAIN_MODEL = None  # 继续训练的模型路径，None则从头训练
    TEST_MODEL = "./tank_ai_models_simple/ppo_gan_simple_ep1200.pth"  # 测试模型路径

    if TRAIN_MODE:
        train_ai(load_model_path=PRETRAIN_MODEL)
    else:
        if not os.path.exists(TEST_MODEL):
            print(f"❌ 测试模型不存在：{TEST_MODEL}")
            print(f"💡 请先执行训练模式生成模型")
        else:
            test_ai(model_path=TEST_MODEL)