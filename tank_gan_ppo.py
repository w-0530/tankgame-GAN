import os
import sys
# 屏蔽macOS Pygame无关系统警告
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pygame
from tqdm import tqdm
from collections import deque
from tankgame import TankGame

# ====================== AI超参数（匹配基础版游戏：8维动作+14维状态） =======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
STATE_DIM = 14  # 和游戏的14维状态严格一致
ACTION_DIM = 8  # 匹配游戏的0-7动作
G_LR = 3e-4     # 生成器学习率
D_LR = 3e-4     # 判别器学习率
GAMMA = 0.99    # PPO折扣因子
LAMBDA = 0.95   # PPO优势函数因子
EPS_CLIP = 0.2  # PPO裁剪系数
BATCH_SIZE = 64 # 训练批次
UPDATE_EPOCH = 10# PPO更新轮数
MEMORY_CAPACITY = 100000  # AI经验池容量
DEMO_MEMORY_CAPACITY = 10000  # 人类演示经验池
TRAIN_EPISODES = 500  # 总训练回合
SAVE_INTERVAL = 50    # 模型保存间隔
RENDER_TRAIN = False  # 训练时关闭渲染提速
RENDER_TEST = True    # 测试时开启渲染
TRAIN_STEP_INTERVAL = 2  # 每N步训练一次

# 创建模型保存文件夹
if not os.path.exists('./tank_ai_models'):
    os.makedirs('./tank_ai_models')

# ====================== GAN网络（纯模型，解耦游戏） =======================
class Generator(nn.Module):
    """生成器：14维状态→8维动作概率分布"""
    def __init__(self, state_dim, action_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

    def get_action(self, state):
        """带探索的动作选择（多项式采样）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_probs = self.forward(state)
        action = torch.multinomial(action_probs, 1).item()
        action_prob = action_probs[0, action].item()
        return action, action_prob

    def get_best_action(self, state):
        """无探索的最优动作选择（贪心）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_probs = self.forward(state)
        return torch.argmax(action_probs, 1).item()

class Discriminator(nn.Module):
    """判别器：14维状态+8维动作→0-1优秀度评分"""
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x))

# ====================== GAN-PPO核心算法（根源解决类型问题） =======================
class GAN_PPO:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.G = Generator(state_dim, action_dim).to(DEVICE)
        self.D = Discriminator(state_dim, action_dim).to(DEVICE)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=G_LR, weight_decay=1e-5)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=D_LR, weight_decay=1e-5)
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.demo_memory = deque(maxlen=DEMO_MEMORY_CAPACITY)
        self.criterion = nn.BCELoss()  # 二分类损失

    def one_hot(self, action):
        """动作转8维float32独热编码"""
        action_one_hot = torch.zeros(len(action), self.action_dim, dtype=torch.float32).to(DEVICE)
        action_one_hot[range(len(action)), action] = 1.0
        return action_one_hot

    def store_memory(self, state, action, action_prob, reward, next_state, done):
        """存储AI试错经验"""
        self.memory.append((state, action, action_prob, reward, next_state, done))

    def store_demo_memory(self, state, action, reward):
        """存储人类演示经验"""
        self.demo_memory.append((state, action, reward))

    def train_D(self):
        """训练判别器：区分人类/AI经验，根源统一float32"""
        if len(self.demo_memory) < BATCH_SIZE//2 or len(self.memory) < BATCH_SIZE//2:
            return 0.0
        
        # 1. 采样人类经验（正样本）- 全程float32，从根源避免double
        demo_idx = np.random.choice(len(self.demo_memory), BATCH_SIZE//2, replace=False)
        demo_data = [self.demo_memory[i] for i in demo_idx]
        demo_s = torch.FloatTensor([d[0] for d in demo_data]).to(DEVICE)
        demo_a = self.one_hot([d[1] for d in demo_data])
        # ✅ 核心修复：用torch生成随机数，直接是float32，彻底抛弃numpy的float64
        demo_l = torch.ones(BATCH_SIZE//2, 1, dtype=torch.float32).to(DEVICE) * torch.FloatTensor(np.random.uniform(0.9, 1.0, (BATCH_SIZE//2, 1))).to(DEVICE)

        # 2. 采样AI经验（负样本）- 全程float32
        ai_idx = np.random.choice(len(self.memory), BATCH_SIZE//2, replace=False)
        ai_data = [self.memory[i] for i in ai_idx]
        ai_s = torch.FloatTensor([d[0] for d in ai_data]).to(DEVICE)
        ai_a = self.one_hot([d[1] for d in ai_data])
        # ✅ 核心修复：同上，torch生成float32随机数
        ai_l = torch.zeros(BATCH_SIZE//2, 1, dtype=torch.float32).to(DEVICE) * torch.FloatTensor(np.random.uniform(0.0, 0.1, (BATCH_SIZE//2, 1))).to(DEVICE)

        # 3. 合并训练
        s = torch.cat([demo_s, ai_s], dim=0)
        a = torch.cat([demo_a, ai_a], dim=0)
        labels = torch.cat([demo_l, ai_l], dim=0)

        self.optimizer_D.zero_grad()
        pred = self.D(s, a)  # 判别器输出天然float32
        loss_D = self.criterion(pred, labels)
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
        self.optimizer_D.step()

        return loss_D.item()

    def compute_gae(self, rewards, dones, values, next_values):
        """计算GAE优势函数"""
        gae = 0
        advantages = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def train_G(self):
        """训练生成器：PPO裁剪优化"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        memory = list(self.memory)
        idx = np.random.choice(len(memory), BATCH_SIZE, replace=False)
        batch = [memory[i] for i in idx]

        # 统一转换为float32张量
        s = torch.FloatTensor([d[0] for d in batch]).to(DEVICE)
        a = torch.LongTensor([d[1] for d in batch]).to(DEVICE)
        old_p = torch.FloatTensor([d[2] for d in batch]).to(DEVICE).unsqueeze(1)
        r = torch.FloatTensor([d[3] for d in batch]).to(DEVICE).unsqueeze(1)
        ns = torch.FloatTensor([d[4] for d in batch]).to(DEVICE)
        done = torch.FloatTensor([d[5] for d in batch]).to(DEVICE).unsqueeze(1)

        # 判别器计算价值
        a_one_hot = self.one_hot(a)
        values = self.D(s, a_one_hot).detach()
        
        # 下一状态价值
        with torch.no_grad():
            next_a_prob = self.G(ns)
            next_a = torch.multinomial(next_a_prob, 1).squeeze()
            next_a_one_hot = self.one_hot(next_a)
            next_values = self.D(ns, next_a_one_hot)

        # GAE优势函数
        advantages = self.compute_gae(
            r.cpu().numpy().squeeze(), 
            done.cpu().numpy().squeeze(),
            values.cpu().numpy().squeeze(), 
            next_values.cpu().numpy().squeeze()
        )

        # PPO训练
        loss_G_total = 0.0
        for _ in range(UPDATE_EPOCH):
            new_p = self.G(s).gather(1, a.unsqueeze(1))
            ratio = torch.exp(torch.log(new_p + 1e-8) - torch.log(old_p + 1e-8))
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantages.unsqueeze(1)
            loss_G = -torch.min(surr1, surr2).mean()

            self.optimizer_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
            self.optimizer_G.step()
            loss_G_total += loss_G.item()

        return loss_G_total / UPDATE_EPOCH

    def save_model(self, episode):
        """保存生成器+判别器模型"""
        torch.save(self.G.state_dict(), f'./tank_ai_models/generator_ep{episode}.pth')
        torch.save(self.D.state_dict(), f'./tank_ai_models/discriminator_ep{episode}.pth')
        print(f"\n模型保存成功：tank_ai_models/ep{episode}")

    def load_model(self, g_path, d_path):
        """加载预训练模型"""
        self.G.load_state_dict(torch.load(g_path, map_location=DEVICE, weights_only=True))
        self.D.load_state_dict(torch.load(d_path, map_location=DEVICE, weights_only=True))
        self.G.eval()
        self.D.eval()
        print("模型加载成功，已进入评估模式")

# ====================== 人类演示数据采集（无冗余逻辑） =======================
def collect_demo_data(game, ppo):
    """采集人类操作数据，WASD移动/方向键转管/空格射击，Q退出"""
    print("="*50)
    print("开始采集人类演示数据！")
    print("操作说明：W(上) A(左) S(下) D(右) | ←→旋转炮管 | 空格射击 | Q退出 | ESC重置")
    print("="*50)
    state = game.reset()
    demo_count = 0
    clock = pygame.time.Clock()
    demo_data = []
    while True:
        clock.tick(60)
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print(f"\n采集完成！共收集 {demo_count} 条人类演示数据")
                    return
                if event.key == pygame.K_ESCAPE:
                    state = game.reset()
                    demo_data = []
                    print("游戏已重置，重新开始采集")

        # 按键映射
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_w]: action = 1
        elif keys[pygame.K_s]: action = 2
        elif keys[pygame.K_a]: action = 3
        elif keys[pygame.K_d]: action = 4
        elif keys[pygame.K_LEFT]: action = 5
        elif keys[pygame.K_RIGHT]: action = 6
        elif keys[pygame.K_SPACE]: action = 7

        # 执行动作并存储数据
        game.do_action(action)
        reward, done = game.step()
        next_state = game.get_state()

        # 过滤连续无操作，只存储有效动作
        if not demo_data or not (action == 0 and demo_data[-1] == 0):
            ppo.store_demo_memory(state, action, reward)
            demo_count += 1
            demo_data.append(action)
            if demo_count % 500 == 0:
                print(f"已采集 {demo_count} 条演示数据 | 最新奖励：{reward:.2f}")

        state = next_state
        if done:
            state = game.reset()
            demo_data = []

# ====================== AI训练主逻辑（采集开窗口/训练关窗口） =======================
def train_ai():
    """完整训练流程：采集人类数据 → GAN-PPO训练 → 保存模型"""
    pygame.init()
    # 采集阶段强制开窗口，保证能操作
    game = TankGame(render=True)
    ppo = GAN_PPO(STATE_DIM, ACTION_DIM)

    # 采集人类演示数据
    collect_demo_data(game, ppo)
    if len(ppo.demo_memory) < 100:
        print("⚠️  警告：演示数据不足100条，训练效果可能较差！")
    
    # 采集完成，关闭渲染提速
    game.render = RENDER_TRAIN

    # 开始GAN-PPO训练
    print(f"\n🚀 开始AI训练！共{TRAIN_EPISODES}回合 | 设备：{DEVICE} | 批次：{BATCH_SIZE}")
    print("="*60)
    clock = pygame.time.Clock()
    for episode in tqdm(range(1, TRAIN_EPISODES+1), desc="AI训练进度", unit="回合"):
        state = game.reset()
        total_reward = 0.0
        total_loss_D = 0.0
        total_loss_G = 0.0
        step = 0

        while True:
            step += 1
            clock.tick(100)
            # 生成器选动作
            action, action_prob = ppo.G.get_action(state)
            game.do_action(action)
            reward, done = game.step()
            next_state = game.get_state()

            # 存储经验
            ppo.store_memory(state, action, action_prob, reward, next_state, done)
            total_reward += reward

            # 间隔训练判别器和生成器
            if step % TRAIN_STEP_INTERVAL == 0:
                loss_D = ppo.train_D()
                loss_G = ppo.train_G()
                total_loss_D += loss_D
                total_loss_G += loss_G

            if done:
                break
            state = next_state

        # 打印每回合训练信息
        avg_loss_D = total_loss_D / step if step > 0 else 0.0
        avg_loss_G = total_loss_G / step if step > 0 else 0.0
        tqdm.write(
            f"回合{episode:3d} | 总奖励{total_reward:6.1f} | D损失{avg_loss_D:.4f} | G损失{avg_loss_G:.4f} | 步数{step:3d}"
        )

        # 按间隔保存模型
        if episode % SAVE_INTERVAL == 0:
            ppo.save_model(episode)

    # 训练完成，保存最终模型
    ppo.save_model(TRAIN_EPISODES)
    pygame.quit()
    print("\n🎉 AI训练完成！所有模型已保存至 ./tank_ai_models 文件夹")

# ====================== AI测试主逻辑（可视化运行） =======================
def test_ai(g_model_path, d_model_path):
    """测试训练好的AI，可视化运行"""
    pygame.init()
    game = TankGame(render=RENDER_TEST)
    ppo = GAN_PPO(STATE_DIM, ACTION_DIM)
    ppo.load_model(g_model_path, d_model_path)

    print("="*50)
    print("开始AI自动玩游戏！按Q或关闭窗口退出")
    print("="*50)
    clock = pygame.time.Clock()
    while True:
        state = game.reset()
        total_reward = 0.0
        step = 0
        while True:
            clock.tick(60)
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    return

            # AI选最优动作执行
            action = ppo.G.get_best_action(state)
            game.do_action(action)
            reward, done = game.step()
            next_state = game.get_state()

            total_reward += reward
            step += 1
            state = next_state

            if done:
                print(f"测试回合结束 | 总奖励：{total_reward:.1f} | 总步数：{step}")
                break

# ====================== 训练/测试入口 =======================
if __name__ == "__main__":
    # 第一步：训练AI（先运行这个，采集数据并训练）
    train_ai()

    # 第二步：测试AI（训练完成后，取消注释并替换模型路径）
    # g_path = "./tank_ai_models/generator_ep500.pth"
    # d_path = "./tank_ai_models/discriminator_ep500.pth"
    # test_ai(g_path, d_path)