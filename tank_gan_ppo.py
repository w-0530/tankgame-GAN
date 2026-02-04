# tankgame_working_train_fixed.py
"""
🚀 坦克游戏AI - 真正有效的工作训练脚本（修复版）
基于诊断结果：规则AI能工作，所以神经网络应该也能学习
修复了Pygame字体初始化问题
"""

import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
from collections import deque
import time
import os

# ============ 初始化 ============
pygame.init()
pygame.font.init()

# ============ 导入游戏 ============
from tankgame import TankGame, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_GUN_LEFT, ACTION_GUN_RIGHT

# ============ 修复：正确的动作映射 ============
# 根据tankgame.py中的实际常量
ACTION_MAP = {
    0: None,              # 无动作
    1: ACTION_UP,         # 上
    2: ACTION_DOWN,       # 下
    3: ACTION_LEFT,       # 左
    4: ACTION_RIGHT,      # 右
    5: ACTION_GUN_LEFT,   # 炮管左转
    6: ACTION_GUN_RIGHT   # 炮管右转
}

# 反向映射：游戏动作 -> 网络动作索引
GAME_ACTION_TO_IDX = {v: k for k, v in ACTION_MAP.items() if v is not None}

# ============ 超参数 ============
STATE_DIM = 14
ACTION_DIM = 7  # 0-6，但0是无动作

# 训练参数
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# ============ 经验回放缓冲区 ============
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# ============ 聪明的神经网络 ============
class SmartAIModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # 使用更深的网络
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 初始化偏向瞄准动作（5,6）
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，偏向瞄准动作"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # 最后层偏向炮管转动
        with torch.no_grad():
            last_layer = self.net[-1]
            last_layer.weight[5] += 0.5  # 炮左转
            last_layer.weight[6] += 0.5  # 炮右转
    
    def forward(self, x):
        return self.net(x)

# ============ DQN Agent ============
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.policy_net = SmartAIModel(state_dim, action_dim).to(self.device)
        self.target_net = SmartAIModel(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # 经验回放
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        # 探索参数
        self.epsilon = EPSILON_START
        self.steps_done = 0
        
        # 训练统计
        self.episode_rewards = []
        self.episode_kills = []
    
    def select_action(self, state, game=None):
        """选择动作，带探索"""
        self.steps_done += 1
        
        # 衰减epsilon
        self.epsilon = max(EPSILON_END, 
                          EPSILON_START * (EPSILON_DECAY ** self.steps_done))
        
        # epsilon-greedy策略
        if random.random() < self.epsilon:
            # 探索：随机选择动作，但偏向有用的动作
            if random.random() < 0.6:  # 60%选择瞄准动作
                return random.choice([5, 6])  # 炮管转动
            else:
                return random.choice([1, 2, 3, 4])  # 移动
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                # 增强瞄准动作的Q值（如果接近敌人）
                if game and game.enemies:
                    enemy = game.enemies[0]
                    dx = enemy.x - game.player.x
                    dy = enemy.y - game.player.y
                    target_angle = math.atan2(-dy, dx)
                    current_angle = game.player.aim_angle
                    
                    angle_diff = abs((target_angle - current_angle) % (2 * math.pi))
                    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                    
                    # 如果没瞄准好，增强瞄准动作
                    if angle_diff > 0.3:  # 大于17度
                        q_values[0, 5] += 2.0  # 炮左转
                        q_values[0, 6] += 2.0  # 炮右转
                
                return q_values.argmax(dim=1).item()
    
    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        # 采样
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # 转为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_kills': self.episode_kills
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_kills = checkpoint['episode_kills']

# ============ 增强的奖励函数 ============
def calculate_reward(game, prev_score, prev_enemies_count):
    """计算增强的奖励"""
    reward = 0.0
    
    # 1. 击杀奖励（最重要）
    current_score = game.score
    if current_score > prev_score:
        kill_reward = 100.0  # 大幅奖励击杀
        reward += kill_reward
    
    # 2. 击中敌人奖励
    # 这里需要根据游戏实际情况调整
    
    # 3. 生存奖励
    reward += 0.1  # 每步生存奖励
    
    # 4. 瞄准质量奖励
    if game.enemies:
        enemy = game.enemies[0]
        dx = enemy.x - game.player.x
        dy = enemy.y - game.player.y
        target_angle = math.atan2(-dy, dx)
        current_angle = game.player.aim_angle
        
        angle_diff = abs((target_angle - current_angle) % (2 * math.pi))
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        
        # 瞄准越好奖励越高
        aim_reward = 0.5 * (1.0 - angle_diff / math.pi)
        reward += aim_reward
    
    # 5. 惩罚被击中
    # 这里需要根据游戏实际情况调整
    
    # 6. 惩罚无效开火
    if game.player.auto_shoot:
        reward -= 0.01  # 轻微惩罚开火消耗
    
    return reward

# ============ 训练循环 ============
def train_dqn():
    print("🚀 开始DQN训练")
    print("=" * 60)
    
    # 创建游戏和智能体
    game = TankGame(render=False)
    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    
    # 训练参数
    num_episodes = 2000
    target_update = 10  # 每10轮更新目标网络
    save_interval = 100  # 每100轮保存模型
    
    # 创建保存目录
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 预填充经验池
    print("预填充经验池...")
    while len(agent.memory) < BATCH_SIZE * 2:
        state = game.reset()
        prev_score = game.score
        
        for step in range(50):
            # 随机动作（探索）
            action = random.choice([1, 2, 3, 4, 5, 6])
            
            # 执行动作
            if action in ACTION_MAP and ACTION_MAP[action]:
                game.do_action(ACTION_MAP[action])
            
            # 自动开火逻辑
            if action in [5, 6] and game.enemies:
                enemy = game.enemies[0]
                dx = enemy.x - game.player.x
                dy = enemy.y - game.player.y
                target_angle = math.atan2(-dy, dx)
                current_angle = game.player.aim_angle
                
                angle_diff = abs((target_angle - current_angle) % (2 * math.pi))
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                
                if angle_diff / math.pi < 0.3:
                    game.player.auto_shoot = True
            
            # 游戏步进
            game.step()
            next_state = game.get_state()
            
            # 计算奖励
            reward = calculate_reward(game, prev_score, 0)
            prev_score = game.score
            
            # 存储经验
            done = game.game_over
            agent.memory.push(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                break
    
    print(f"经验池预填充完成: {len(agent.memory)} 条经验")
    
    # 主训练循环
    print("\n开始主训练循环...")
    
    for episode in range(num_episodes):
        # 重置环境
        state = game.reset()
        prev_score = game.score
        episode_reward = 0
        episode_kills = 0
        
        # 获取初始击杀数
        initial_kills = game.score // 70 if hasattr(game, 'score') else 0
        
        for step in range(300):  # 每回合最多300步
            # 选择动作
            action = agent.select_action(state, game)
            
            # 执行动作
            if action in ACTION_MAP and ACTION_MAP[action]:
                game.do_action(ACTION_MAP[action])
            
            # 智能自动开火
            if game.enemies and action in [5, 6]:
                enemy = game.enemies[0]
                dx = enemy.x - game.player.x
                dy = enemy.y - game.player.y
                target_angle = math.atan2(-dy, dx)
                current_angle = game.player.aim_angle
                
                angle_diff = abs((target_angle - current_angle) % (2 * math.pi))
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                
                # 只有瞄准得比较好时才开火
                if angle_diff / math.pi < 0.2:  # 20%误差内
                    game.player.auto_shoot = True
                else:
                    game.player.auto_shoot = False
            else:
                game.player.auto_shoot = False
            
            # 游戏步进
            game.step()
            next_state = game.get_state()
            
            # 计算奖励
            reward = calculate_reward(game, prev_score, 0)
            prev_score = game.score
            
            # 统计击杀
            current_kills = game.score // 70 if hasattr(game, 'score') else 0
            if current_kills > episode_kills:
                episode_kills = current_kills
            
            # 存储经验
            done = game.game_over or step == 299
            agent.memory.push(state, action, reward, next_state, done)
            
            # 优化模型
            loss = agent.optimize_model()
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 记录统计
        agent.episode_rewards.append(episode_reward)
        agent.episode_kills.append(episode_kills)
        
        # 更新目标网络
        if episode % target_update == 0:
            agent.update_target_net()
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(agent.episode_rewards[-50:])
            avg_kills = np.mean(agent.episode_kills[-50:])
            kill_rate = sum(agent.episode_kills[-50:]) / 50 * 100
            
            print(f"回合 {episode+1:4d} | "
                  f"平均奖励: {avg_reward:6.1f} | "
                  f"平均击杀: {avg_kills:4.1f} | "
                  f"击杀率: {kill_rate:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent.save_model(f"./checkpoints/model_episode_{episode+1}.pth")
    
    # 保存最终模型
    agent.save_model("./tank_ai_final.pth")
    print("\n✅ 训练完成！模型已保存为 tank_ai_final.pth")
    
    pygame.quit()
    return agent

# ============ 测试训练好的模型 ============
def test_trained_model(model_path=None):
    """测试训练好的模型（修复版：确保Pygame字体初始化）"""
    print("\n🧪 测试AI性能")
    print("=" * 60)
    
    # 🚨 修复：确保Pygame和字体模块已初始化
    try:
        pygame.init()
        pygame.font.init()
    except:
        pass
    
    game = TankGame(render=True)
    
    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        agent = DQNAgent(STATE_DIM, ACTION_DIM)
        agent.load_model(model_path)
        agent.epsilon = 0.01  # 测试时用很小的探索率
    else:
        print("使用新模型")
        agent = DQNAgent(STATE_DIM, ACTION_DIM)
    
    num_test_episodes = 10
    total_kills = 0
    total_steps = 0
    
    for episode in range(num_test_episodes):
        state = game.reset()
        episode_kills = 0
        episode_steps = 0
        
        print(f"\n测试回合 {episode+1}:")
        
        for step in range(200):
            # 选择动作
            action = agent.select_action(state, game)
            
            # 执行动作
            if action in ACTION_MAP and ACTION_MAP[action]:
                game.do_action(ACTION_MAP[action])
            
            # 自动开火
            if action in [5, 6]:
                game.player.auto_shoot = True
            
            # 游戏步进
            game.step()
            next_state = game.get_state()
            
            # 检查击杀
            kills = game.score // 70 if hasattr(game, 'score') else 0
            if kills > episode_kills:
                episode_kills = kills
                print(f"  步{step}: 击杀！")
            
            # 更新
            state = next_state
            episode_steps += 1
            
            if game.game_over:
                break
            
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        total_kills += episode_kills
        total_steps += episode_steps
        
        print(f"  结束: 击杀={episode_kills}, 步数={episode_steps}")
    
    pygame.quit()
    
    avg_kills = total_kills / num_test_episodes
    avg_steps = total_steps / num_test_episodes
    
    print(f"\n📊 测试结果:")
    print(f"  平均每回合击杀: {avg_kills:.2f}")
    print(f"  平均每回合步数: {avg_steps:.2f}")
    
    if avg_kills > 0.5:
        print("✅ AI学习成功！")
    else:
        print("⚠️  AI仍需改进")

# ============ 主函数 ============
def main():
    print("🎯 坦克游戏AI - 工作训练脚本（修复版）")
    print("=" * 60)
    print("基于诊断结果设计:")
    print("1. 规则AI能击杀 → 神经网络应该也能学习")
    print("2. 增强奖励函数，明确反馈")
    print("3. 使用更深的网络结构")
    print("4. 经验回放 + 目标网络")
    print("5. 修复了Pygame字体初始化问题")
    print("=" * 60)
    
    while True:
        print("\n选项:")
        print("1. 开始训练DQN")
        print("2. 测试现有模型")
        print("3. 快速测试（只运行规则AI）")
        print("4. 退出")
        
        choice = input("请选择 (1-4): ").strip()
        
        if choice == "1":
            print("\n开始训练...")
            agent = train_dqn()
            
            # 训练后立即测试
            test_trained_model("./tank_ai_final.pth")
            
        elif choice == "2":
            model_path = input("模型路径 (默认: ./tank_ai_final.pth): ").strip()
            if not model_path:
                model_path = "./tank_ai_final.pth"
            test_trained_model(model_path)
            
        elif choice == "3":
            print("\n运行规则AI测试...")
            # 确保Pygame初始化
            try:
                pygame.init()
                pygame.font.init()
            except:
                pass
            
            # 使用之前的规则AI测试
            from tankgame import TankGame
            import random
            import math
            
            game = TankGame(render=True)
            
            test_episodes = 5
            total_kills = 0
            
            for ep in range(test_episodes):
                state = game.reset()
                kills = 0
                
                print(f"\n回合 {ep+1}:")
                
                for step in range(200):
                    if game.game_over:
                        break
                    
                    # 规则AI逻辑
                    if game.enemies:
                        enemy = game.enemies[0]
                        dx = enemy.x - game.player.x
                        dy = enemy.y - game.player.y
                        target_angle = math.atan2(-dy, dx)
                        current_angle = game.player.aim_angle
                        
                        angle_diff = (target_angle - current_angle) % (2 * math.pi)
                        if angle_diff > math.pi:
                            angle_diff -= 2 * math.pi
                        
                        if angle_diff > 0.1:
                            action = ACTION_GUN_LEFT
                        elif angle_diff < -0.1:
                            action = ACTION_GUN_RIGHT
                        else:
                            game.player.auto_shoot = True
                            action = random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])
                    else:
                        action = random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])
                    
                    # 执行动作
                    game.do_action(action)
                    game.step()
                    
                    # 检查击杀
                    current_kills = game.score // 70
                    if current_kills > kills:
                        kills = current_kills
                        print(f"  步{step}: 击杀！总击杀{kills}")
                    
                    # 处理退出事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                
                total_kills += kills
                print(f"  回合结束: 击杀{kills}")
            
            pygame.quit()
            
            avg_kills = total_kills / test_episodes
            print(f"\n📊 规则AI测试: 平均每回合{avg_kills:.1f}击杀")
            
        elif choice == "4":
            print("👋 退出")
            break
        
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    # 检查依赖
    try:
        import torch
        main()
    except ImportError:
        print("❌ 需要安装PyTorch: pip install torch")
        print("运行快速测试...")
        
        # 确保Pygame初始化
        try:
            pygame.init()
            pygame.font.init()
        except:
            pass
        
        # 运行不需要PyTorch的测试
        test_episodes = 3
        total_kills = 0
        
        for ep in range(test_episodes):
            game = TankGame(render=True)
            state = game.reset()
            kills = 0
            
            for step in range(200):
                if game.game_over:
                    break
                
                # 随机动作
                action = random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, 
                                       ACTION_GUN_LEFT, ACTION_GUN_RIGHT])
                game.do_action(action)
                
                # 随机开火
                if random.random() < 0.3:
                    game.player.auto_shoot = True
                
                game.step()
                
                # 检查击杀
                current_kills = game.score // 70
                if current_kills > kills:
                    kills = current_kills
                    print(f"回合{ep+1} 步{step}: 击杀！")
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
            
            total_kills += kills
            pygame.quit()
        
        avg_kills = total_kills / test_episodes
        print(f"\n随机AI平均每回合击杀: {avg_kills:.1f}")