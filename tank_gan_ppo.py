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

# ====================== 基础配置 =======================
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

# 设备自动适配
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 训练设备：{DEVICE} | CUDA可用：{torch.cuda.is_available()}")

# 核心维度定义
STATE_DIM = 14
ACTION_DIM = 8
DEMO_VEC_DIM = STATE_DIM + ACTION_DIM

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

# ====================== GAN 超参数 ======================
GAN_LR = 4e-5
GAN_UPDATE_INTERVAL = 5
GAN_REWARD_WEIGHT = 0.1  # 核心调整：0.3→0.1
GAN_TRAIN_TIMES = 1
GAN_STOP_EPOCH = 1200
GAN_BCE_WEIGHT = 0.5

# ====================== 奖励函数核心参数（集中定义）======================
# 基础奖励：拉满击杀激励，风险收益比10倍，击杀是唯一正收益来源
KILL_REWARD = 300.0     # 保持你的300，一次击杀覆盖所有惩罚
HIT_REWARD = 10.0       # 不变，次要奖励
BEEN_HIT_PUNISH = 30.0  # 不变，不提高，避免模型重回极端保守
DEATH_PUNISH = 50.0     # 不变
# 未击杀惩罚：大幅强化+缩短步数，让躺平避战必亏分，制造强击杀紧迫感
NO_KILL_PUNISH = 15.0   # 从9→15，惩罚翻倍+，每触发一次亏分显著
NO_KILL_STEP_THRESH = 20# 从70→20，核心修改！每20步未击杀就扣15，350步躺平会扣225，直接亏麻
# 行为奖惩：砍掉躺平基础奖+拉满避战惩罚，逼模型开火/找敌人
MOVE_ACTION_REW = 0.05  # 从0.2→0.05，几乎无移动奖，不让模型靠走步躺平
OPTIMAL_DIST_REW = 0.3  # 从1.2→0.3，仅保留微量控距奖（避免贴脸/远距）
MOVE_TOWARD_ENEMY_REW = 1.5 # 从0.8→1.5，唯一强化的奖励：主动向敌人移动才给奖，逼模型找敌人
AIM_ACTION_REW = 0.5    # 从0.3→0.5，强化有效瞄准奖励，鼓励模型对准敌人
SINGLE_ACTION_PUNISH = 0.5 # 不变，重复动作惩罚
CLOSE_NO_SHOOT_PUNISH = 5.0# 从1→5，核心修改！近距离（<200）不开火每步扣5，近距离必须开火
IDLE_PUNISH = 1.0       # 从0.2→1.0，核心修改！原地不动每步扣1，逼模型动起来找敌人
# 距离参数：不变，保持原有控距范围
OPTIMAL_DIST_LOW = 150
OPTIMAL_DIST_HIGH = 350
CLOSE_DIST_THRESHOLD = 200
DIST_JUDGE_THRESHOLD = 5
# 其他参数：放宽开火试错，让模型敢开火、不怕失误
MAX_ANGLE_ERROR = math.pi/6 # 从math.pi→π/6，核心修改！缩小有效瞄准角度（30°内），让瞄准更精准才给奖
MAX_INVALID_FIRE = 5    # 从3→5，连续5次无效开火才无奖励，给足试错空间
MAX_STEP = 350
NO_COMBAT_STEP = 20     # 从45→20，缩短无战斗步数，逼模型快速接战
# ====================== 训练配置 =======================
MEMORY_CAPACITY = 60000
DEMO_MEMORY_CAPACITY = 4000
TRAIN_EPISODES = 1200
SAVE_INTERVAL = 100
RENDER_TRAIN = False
RENDER_TEST = True
TRAIN_STEP_INTERVAL = 3
DEMO_SAMPLE_RATIO = 0.5

# 创建保存目录
os.makedirs("./tank_ai_models", exist_ok=True)
os.makedirs("./tank_demo_data", exist_ok=True)

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
        assert demo_vec.shape[0] == self.vec_dim, f"维度错误：{demo_vec.shape[0]} vs {self.vec_dim}"
        self.memory.append(demo_vec)
    
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        demo_array = np.array([self.memory[i] for i in idx], dtype=np.float32)
        return torch.FloatTensor(demo_array).to(DEVICE)
    
    def load_from_npy(self, npy_path):
        demo_array = np.load(npy_path, dtype=np.float32)
        assert demo_array.shape[1] == self.vec_dim, f"npy维度错误：{demo_array.shape[1]} vs {self.vec_dim}"
        for vec in demo_array:
            self.add(vec)
        print(f"📚 加载专家经验：{len(demo_array)}条 | 维度：{self.vec_dim}")
    
    def save_to_npy(self, npy_path):
        demo_array = np.array(self.memory, dtype=np.float32)
        np.save(npy_path, demo_array)
        print(f"💾 保存专家经验：{len(demo_array)}条 | 路径：{npy_path}")
    
    def __len__(self):
        return len(self.memory)

# =========================================================
# PPO 网络结构
# =========================================================
class PPO_Actor(nn.Module):
    def __init__(self, state_dim, action_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, action_dim)
        ).to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        return F.softmax(self.net(x), dim=-1)

    def get_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self(s)
        a = torch.multinomial(probs, 1).item()
        return a, probs[0, a].item()

    def get_best_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
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
        ).to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        return self.net(x)

# =========================================================
# GAN 判别器
# =========================================================
class GAILGAN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 1)
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=GAN_LR)
    
    def forward(self, x):
        x = x.to(DEVICE)
        return self.discriminator(x)
    
    def train_step(self, agent_batch, expert_batch):
        expert_logits = self(expert_batch)
        agent_logits = self(agent_batch)

        loss_expert = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits).to(DEVICE)
        )
        loss_agent = F.binary_cross_entropy_with_logits(
            agent_logits, torch.zeros_like(agent_logits).to(DEVICE)
        )
        loss = GAN_BCE_WEIGHT * loss_expert + (1 - GAN_BCE_WEIGHT) * loss_agent

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            gan_reward = -torch.log(torch.sigmoid(agent_logits) + 1e-8).mean().item()

        return loss.item(), gan_reward

# =========================================================
# PPO-GAN 智能体
# =========================================================
class PPO_GAN:
    def __init__(self, state_dim=14, action_dim=8):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = PPO_Actor(state_dim, action_dim)
        self.critic = PPO_Critic(state_dim)

        self.ppo_opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=PPO_LR
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.ppo_opt, step_size=PPO_LR_DECAY_EPOCH, gamma=PPO_LR_DECAY_RATE
        )

        self.ppo_memory = PPOMemory(MEMORY_CAPACITY)
        self.demo_memory = DemoMemory(DEMO_MEMORY_CAPACITY, DEMO_VEC_DIM)
        self.gan = GAILGAN(DEMO_VEC_DIM)

        self.reset_combat_state()

    def reset_combat_state(self):
        self.player_lives_cache = 0
        self.last_score = 0
        self.last_enemy_dist = float("inf")
        self.last_action = -1
        self.been_hit_flag = False
        self.no_combat_step = 0
        self.invalid_fire_count = 0
        self.no_kill_step = 0

    def one_hot(self, actions):
        oh = torch.zeros(len(actions), self.action_dim).to(DEVICE)
        oh[range(len(actions)), actions] = 1.0
        return oh
    
    def compute_gae(self, r, d, v, nv):
        gae = 0
        adv = torch.zeros_like(r).to(DEVICE)
        for i in reversed(range(len(r))):
            delta = r[i] + GAMMA * nv[i] * (1 - d[i]) - v[i]
            gae = delta + GAMMA * LAMBDA * (1 - d[i]) * gae
            adv[i] = gae
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return torch.clamp(adv, -ADV_CLIP, ADV_CLIP)

    def train_ppo(self):
        batch = self.ppo_memory.sample(BATCH_SIZE)
        if batch is None:
            return 0.0, 0.0, 0.0
        s, a, r, ns, d, old_p = batch

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
        total_loss = actor_loss + 0.5 * critic_loss - ENT_COEF * entropy

        self.ppo_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), 1.2
        )
        self.ppo_opt.step()
        self.lr_scheduler.step()

        return total_loss.item(), actor_loss.item(), critic_loss.item()

    def save(self, ep):
        save_path = f"./tank_ai_models/ppo_gan_fire_ep{ep}.pth"
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "gan": self.gan.state_dict(),
            "ppo_opt": self.ppo_opt.state_dict(),
            "gan_opt": self.gan.optimizer.state_dict(),
            "epoch": ep
        }, save_path)
        print(f"\n💾 模型保存：{save_path}")

    def load(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在：{model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.gan.load_state_dict(checkpoint["gan"])
        self.ppo_opt.load_state_dict(checkpoint["ppo_opt"])
        self.gan.optimizer.load_state_dict(checkpoint["gan_opt"])
        print(f"✅ 加载模型：{model_path}")

# =========================================================
# 工具函数（添加global声明，解决Pylance提示）
# =========================================================
def get_nearest_enemy(game):
    enemies_alive = [e for e in game.enemies if e.alive]
    if not enemies_alive:
        return None
    distances = [math.hypot(e.x - game.player.x, e.y - game.player.y) for e in enemies_alive]
    return enemies_alive[np.argmin(distances)]

def calculate_enemy_dist(game):
    enemy = get_nearest_enemy(game)
    return math.hypot(enemy.x - game.player.x, enemy.y - game.player.y) if enemy else float("inf")

def calculate_aim_error(game):
    global MAX_ANGLE_ERROR
    enemy = get_nearest_enemy(game)
    if not enemy:
        return 0.0, 0.0
    
    player = game.player
    dx = enemy.x - player.x
    dy = enemy.y - player.y
    target_angle = math.atan2(-dy, dx) % (2 * math.pi)
    current_angle = player.aim_angle % (2 * math.pi)

    angle_error = abs(current_angle - target_angle)
    angle_error = min(angle_error, 2 * math.pi - angle_error)
    face_coeff = math.cos(angle_error) if angle_error <= math.pi/2 else 0.0
    aim_coeff = 1.0 - min(angle_error / MAX_ANGLE_ERROR, 1.0)

    return face_coeff, aim_coeff

def check_reward_trigger(game, agent):
    global KILL_REWARD, HIT_REWARD
    is_kill, is_hit, is_hurt, is_been_hit = False, False, False, False

    if agent.player_lives_cache == 0:
        agent.player_lives_cache = game.player.lives

    if game.score > agent.last_score:
        score_diff = game.score - agent.last_score
        is_kill = (score_diff == KILL_REWARD)
        is_hit = (score_diff > 0 and not is_kill)
        agent.last_score = game.score
        agent.invalid_fire_count = 0
        agent.no_kill_step = 0

    if game.player.lives < agent.player_lives_cache:
        is_been_hit = True
        agent.been_hit_flag = True
        agent.player_lives_cache = game.player.lives

    is_hurt = game.player.lives < agent.player_lives_cache if agent.player_lives_cache else False
    return is_kill, is_hit, is_hurt, is_been_hit

# =========================================================
# 奖励函数（全量global声明，解决Pylance提示）
# =========================================================
def get_env_reward(game, action, step, agent):
    # 声明所有使用的全局变量，彻底解决Pylance未定义提示
    global KILL_REWARD, HIT_REWARD, BEEN_HIT_PUNISH, DEATH_PUNISH
    global NO_KILL_PUNISH, NO_KILL_STEP_THRESH, MOVE_ACTION_REW
    global OPTIMAL_DIST_REW, OPTIMAL_DIST_LOW, OPTIMAL_DIST_HIGH
    global SURVIVE_REWARD, FACE_ENEMY_REWARD, EFFECTIVE_AIM_REWARD
    global MOVE_TOWARD_ENEMY_REW, AIM_ACTION_REW, SINGLE_ACTION_PUNISH
    global CLOSE_DIST_THRESHOLD, CLOSE_NO_SHOOT_PUNISH, IDLE_PUNISH
    global DIST_JUDGE_THRESHOLD, MAX_INVALID_FIRE, NO_COMBAT_STEP

    final_reward = 0.0
    is_combat = False
    # 动作常量
    ACTION_IDLE, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0,1,2,3,4
    ACTION_GUN_LEFT, ACTION_GUN_RIGHT, ACTION_SHOOT =5,6,7
    move_actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
    aim_actions = [ACTION_GUN_LEFT, ACTION_GUN_RIGHT]
    all_actions = move_actions + aim_actions + [ACTION_SHOOT]

    # 核心状态
    fired = (action == ACTION_SHOOT)
    enemy = get_nearest_enemy(game)
    enemy_visible = (enemy is not None and enemy.alive and game.player.alive)
    current_dist = calculate_enemy_dist(game)

    # 1. 移动加分
    if action in move_actions and enemy_visible:
        final_reward += MOVE_ACTION_REW
        is_combat = True

    # 2. 最优距离加分
    if enemy_visible and OPTIMAL_DIST_LOW <= current_dist <= OPTIMAL_DIST_HIGH:
        final_reward += OPTIMAL_DIST_REW
        is_combat = True

    # 3. 未击杀惩罚
    if enemy_visible:
        agent.no_kill_step += 1
        if agent.no_kill_step >= NO_KILL_STEP_THRESH:
            final_reward -= NO_KILL_PUNISH
            agent.no_kill_step = NO_KILL_STEP_THRESH - 10
    else:
        agent.no_kill_step = 0

    # 奖励密集化
    face_coeff, aim_coeff = calculate_aim_error(game)
    final_reward += 0.01  # SURVIVE_REWARD
    final_reward += 0.02 * face_coeff  # FACE_ENEMY_REWARD
    final_reward += 0.08 * aim_coeff  # EFFECTIVE_AIM_REWARD

    # 向敌人移动额外奖励
    if action in move_actions and enemy_visible:
        if current_dist < agent.last_enemy_dist - DIST_JUDGE_THRESHOLD:
            final_reward += MOVE_TOWARD_ENEMY_REW
            agent.last_enemy_dist = current_dist

    # 瞄准动作奖励
    if action in aim_actions and enemy_visible:
        final_reward += AIM_ACTION_REW
        is_combat = True

    # 重复动作惩罚
    if agent.last_action == action and action in all_actions and step > 5:
        final_reward -= SINGLE_ACTION_PUNISH
    agent.last_action = action

    # 近距离不射击惩罚
    if current_dist < CLOSE_DIST_THRESHOLD and action != ACTION_SHOOT and enemy_visible:
        final_reward -= CLOSE_NO_SHOOT_PUNISH

    # 闲置惩罚
    if action == ACTION_IDLE and enemy_visible:
        final_reward -= IDLE_PUNISH

    # 核心奖励/惩罚
    is_kill, is_hit, is_hurt, is_been_hit = check_reward_trigger(game, agent)
    if is_kill:
        final_reward += KILL_REWARD
        is_combat = True
    if is_hit:
        final_reward += HIT_REWARD
        is_combat = True
    if is_been_hit and agent.been_hit_flag:
        final_reward -= BEEN_HIT_PUNISH
        agent.been_hit_flag = False
        is_combat = True
    if is_hurt:
        final_reward -= DEATH_PUNISH
        is_combat = True

    # 开火规则
    if not fired and enemy_visible:
        final_reward -= 0.2
    if fired:
        is_combat = True
        if agent.invalid_fire_count < MAX_INVALID_FIRE:
            final_reward += 0.05
        agent.invalid_fire_count += 1

    # 战斗状态更新
    agent.no_combat_step = 0 if is_combat else agent.no_combat_step + 1

    return np.clip(final_reward, -80, 150), is_combat

# =========================================================
# 生成专家经验
# =========================================================
def generate_demo_data(demo_memory, demo_num=1000):
    global DEMO_VEC_DIM, ACTION_DIM
    from tankgame import TankGame
    print("\n🎮 生成专家经验 | 按键：W/A/S/D移动 | ←/→转炮管 | 空格射击 | ESC退出")
    print(f"🎯 目标：{demo_num}条 | 维度：{DEMO_VEC_DIM}")

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

        # 手动操作
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: action = 1
        elif keys[pygame.K_s]: action = 2
        elif keys[pygame.K_a]: action = 3
        elif keys[pygame.K_d]: action = 4
        elif keys[pygame.K_LEFT]: action = 5
        elif keys[pygame.K_RIGHT]: action = 6
        elif keys[pygame.K_SPACE]: action = 7

        # 仅保留有效战斗状态
        enemy = get_nearest_enemy(game)
        if game.player.alive and enemy and enemy.alive:
            state_np = np.asarray(state, dtype=np.float32).flatten()
            action_one_hot = np.eye(ACTION_DIM, dtype=np.float32)[action]
            demo_vec = np.concatenate([state_np, action_one_hot])
            demo_memory.add(demo_vec)
            if len(demo_memory) % 100 == 0:
                print(f"📈 采集：{len(demo_memory)}/{demo_num}")

        # 游戏步进
        game.do_action(action)
        game.step()
        state = game.get_state()
        if game.game_over:
            state = game.reset()
            print(f"🔄 游戏重置 | 已采集：{len(demo_memory)}条")

    demo_memory.save_to_npy("./tank_demo_data/demo_memory.npy")
    pygame.quit()
    print(f"\n✅ 专家经验生成完成：{len(demo_memory)}条")

# =========================================================
# 训练入口（添加global声明，解决GAN_REWARD_WEIGHT提示）
# =========================================================
def train_ai(load_model_path=None):
    # 声明训练中使用的所有全局参数
    global GAN_REWARD_WEIGHT, GAN_UPDATE_INTERVAL, BATCH_SIZE
    global STATE_DIM, ACTION_DIM, TRAIN_EPISODES, MAX_STEP
    global NO_COMBAT_STEP, TRAIN_STEP_INTERVAL, SAVE_INTERVAL
    global RENDER_TRAIN, KILL_REWARD, BEEN_HIT_PUNISH
    
    from tankgame import TankGame
    pygame.init()
    game = TankGame(render=RENDER_TRAIN)
    agent = PPO_GAN(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    # 加载预训练模型
    if load_model_path and os.path.exists(load_model_path):
        agent.load(load_model_path)

    # 加载/生成专家经验
    demo_file_path = "./tank_demo_data/demo_memory.npy"
    if os.path.exists(demo_file_path):
        try:
            agent.demo_memory.load_from_npy(demo_file_path)
        except Exception as e:
            print(f"⚠️  加载专家经验失败：{e} | 重新采集")
            generate_demo_data(agent.demo_memory, demo_num=1000)
    else:
        print(f"⚠️  无专家经验 | 开始采集")
        generate_demo_data(agent.demo_memory, demo_num=1000)

    # 经验不足补采集
    if len(agent.demo_memory) < 500:
        generate_demo_data(agent.demo_memory, demo_num=1000)

    # 维度校验
    assert len(game.get_state()) == STATE_DIM, f"状态维度不匹配：{len(game.get_state())} vs {STATE_DIM}"
    print(f"✅ 开始训练 | 总轮数：{TRAIN_EPISODES}")
    print(f"📌 核心配置：GAN权重{GAN_REWARD_WEIGHT} | 击杀奖励{KILL_REWARD} | 未击杀{NO_KILL_STEP_THRESH}步扣{NO_KILL_PUNISH}")
    print(f"📌 风险收益比：击杀+{KILL_REWARD} | 被击中-{BEEN_HIT_PUNISH}（{KILL_REWARD/BEEN_HIT_PUNISH:.1f}倍）")

    # 训练主循环
    pbar = tqdm(range(1, TRAIN_EPISODES + 1), desc="训练进度")
    last_gan_loss = 0.0

    for ep in pbar:
        state = game.reset()
        agent.reset_combat_state()
        agent.player_lives_cache = game.player.lives

        total_reward = 0.0
        done = False
        step = 0
        ppo_loss_sum = 0.0
        gan_reward_sum = 0.0
        ppo_train_count = 0

        while not done:
            step += 1
            if step >= MAX_STEP or agent.no_combat_step >= NO_COMBAT_STEP:
                break

            # PPO选动作
            action, action_prob = agent.actor.get_action(state)
            # 执行动作
            game.do_action(action)
            # 游戏步进
            _, game_done = game.step()
            # 计算奖励
            env_reward, _ = get_env_reward(game, action, step, agent)
            # 下一状态
            next_state = game.get_state()

            # GAN训练与奖励计算
            gan_reward = 0.0
            if step % GAN_UPDATE_INTERVAL == 0 and len(agent.ppo_memory) >= BATCH_SIZE:
                ppo_batch = agent.ppo_memory.sample(BATCH_SIZE)
                s_ppo, a_ppo, _, _, _, _ = ppo_batch
                a_ppo_oh = agent.one_hot(a_ppo.squeeze(1))
                agent_batch = torch.cat([s_ppo, a_ppo_oh], dim=-1)
                expert_batch = agent.demo_memory.sample(BATCH_SIZE)
                gan_loss, gan_reward = agent.gan.train_step(agent_batch, expert_batch)
                last_gan_loss = gan_loss
                gan_reward_sum += gan_reward

            # 最终奖励融合
            final_reward = env_reward + GAN_REWARD_WEIGHT * gan_reward
            agent.ppo_memory.add(state, action, final_reward, next_state, game_done, action_prob)
            total_reward += final_reward

            # 训练PPO
            if step % TRAIN_STEP_INTERVAL == 0:
                ppo_loss, _, _ = agent.train_ppo()
                ppo_loss_sum += ppo_loss
                ppo_train_count += 1

            # 状态更新
            state = next_state
            done = game_done

        # 进度展示
        avg_ppo_loss = ppo_loss_sum / max(ppo_train_count, 1)
        pbar.set_postfix({
            "总奖励": f"{total_reward:.2f}",
            "PPO损失": f"{avg_ppo_loss:.4f}",
            "GAN损失": f"{last_gan_loss:.4f}",
            "经验池": f"{len(agent.ppo_memory)}/{MEMORY_CAPACITY}"
        })

        # 保存模型
        if ep % SAVE_INTERVAL == 0:
            agent.save(ep)

    # 保存最终模型
    agent.save(TRAIN_EPISODES)
    pygame.quit()
    print(f"\n🎉 训练完成！最终模型：./tank_ai_models/ppo_gan_fire_ep{TRAIN_EPISODES}.pth")

# =========================================================
# 测试入口（添加global声明，解决Pylance提示）
# =========================================================
def test_ai(model_path):
    global STATE_DIM, ACTION_DIM, MAX_STEP, RENDER_TEST
    global KILL_REWARD, GAN_REWARD_WEIGHT
    from tankgame import TankGame
    pygame.init()
    game = TankGame(render=RENDER_TEST)
    agent = PPO_GAN(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    agent.load(model_path)

    # 维度校验
    state = game.reset()
    assert len(state) == STATE_DIM, f"状态维度不匹配：{len(state)} vs {STATE_DIM}"
    print(f"\n🎮 AI测试开始 | 模型：{model_path}")
    print(f"📌 配置：击杀奖励{KILL_REWARD} | GAN权重{GAN_REWARD_WEIGHT}")

    clock = pygame.time.Clock()
    total_reward = 0.0
    step = 0
    done = False
    agent.reset_combat_state()
    agent.player_lives_cache = game.player.lives

    while not done:
        step += 1
        clock.tick(60)
        # 取最优动作
        action = agent.actor.get_best_action(state)
        # 执行动作
        game.do_action(action)
        # 游戏步进
        _, game_done = game.step()
        # 计算奖励
        reward, _ = get_env_reward(game, action, step, agent)
        total_reward += reward
        # 下一状态
        state = game.get_state()
        # 结束条件
        done = game_done or step >= MAX_STEP

        # 退出测试
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                return

    # 测试结果
    pygame.quit()
    kill_num = game.score // int(KILL_REWARD) if game.score >= KILL_REWARD else 0
    print(f"✅ 测试结束 | 步数：{step} | 总奖励：{total_reward:.2f}")
    print(f"📊 战绩：得分{game.score} | 击杀{kill_num} | 剩余生命{game.player.lives}")
    print(f"🏆 结果：{'胜利' if kill_num >=8 else '失败'}（胜利条件：击杀≥8）")

# =========================================================
# 主函数（训练/测试一键切换）
# =========================================================
if __name__ == "__main__":
    TRAIN_MODE = True  # True=训练，False=测试
    PRETRAIN_MODEL = None  # 继续训练的模型路径
    TEST_MODEL = "./tank_ai_models/ppo_gan_fire_ep1200.pth"  # 测试模型路径

    if TRAIN_MODE:
        train_ai(load_model_path=PRETRAIN_MODEL)
    else:
        test_ai(model_path=TEST_MODEL)