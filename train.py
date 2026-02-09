import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
import tankgame

# 简单的优先级经验回放实现
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, movement_action, aim_action, reward, next_state, done):
        max_priority = self.max_priority
        experience = (state, movement_action, aim_action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if buffer_len == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 重要性采样权重
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (buffer_len * probs) ** (-self.beta)
        weights /= weights.max()
        
        # 采样
        indices = np.random.choice(buffer_len, batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        weights = torch.FloatTensor(weights[indices])
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # 避免0优先级
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return len(self.buffer)

# 导入游戏常量用于反归一化
SCREEN_WIDTH = tankgame.SCREEN_WIDTH
SCREEN_HEIGHT = tankgame.SCREEN_HEIGHT

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(33, 512),  # 更大网络处理增强状态
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # 移动头：5个动作（静止、上、下、左、右）
        self.movement_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        # 瞄准头：3个动作（左转、右转、射击）
        self.aim_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        movement_q = self.movement_head(shared_features)
        aim_q = self.aim_head(shared_features)
        return movement_q, aim_q

# 动作映射常量
MOVEMENT_ACTIONS = {
    0: tankgame.ACTION_IDLE,    # 静止
    1: tankgame.ACTION_UP,      # 上
    2: tankgame.ACTION_DOWN,    # 下
    3: tankgame.ACTION_LEFT,    # 左
    4: tankgame.ACTION_RIGHT    # 右
}

AIM_ACTIONS = {
    0: tankgame.ACTION_GUN_LEFT,  # 左转
    1: tankgame.ACTION_GUN_RIGHT, # 右转
    2: tankgame.ACTION_SHOOT      # 射击
}

class SimpleAgent:
    def __init__(self):
        self.model = SimpleNet()
        self.target_model = SimpleNet()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        # 添加学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50, verbose=True
        )
        # 使用优先级经验回放
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        self.epsilon = 0.5
        self.gamma = 0.99
        self.batch_size = 512  # 增大批次大小
        self.update_target_freq = 1000
        self.train_count = 0
        
        # 新增训练稳定性参数
        self.max_grad_norm = 1.0
        self.target_update_tau = 0.005  # 软更新参数
        self.use_soft_update = False  # 是否使用软更新
        

    
    def remember(self, state, movement_action, aim_action, reward, next_state, done):
        self.memory.add(state, movement_action, aim_action, reward, next_state, done)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 使用优先级经验回放采样
        batch, indices, weights = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([b[0] for b in batch])
        movement_actions = torch.LongTensor([b[1] for b in batch])
        aim_actions = torch.LongTensor([b[2] for b in batch])
        rewards = torch.FloatTensor([b[3] for b in batch])
        next_states = torch.FloatTensor([b[4] for b in batch])
        dones = torch.BoolTensor([b[5] for b in batch])
        weights = weights.to(states.device)  # 移动到正确设备
        
        # 获取当前Q值
        current_movement_q, current_aim_q = self.model(states)
        current_movement_q = current_movement_q.gather(1, movement_actions.unsqueeze(1))
        current_aim_q = current_aim_q.gather(1, aim_actions.unsqueeze(1))
        
        # 获取下一状态的最大Q值（使用目标网络和Double DQN）
        with torch.no_grad():
            # Double DQN：用主网络选择动作，目标网络评估
            next_movement_q_online, next_aim_q_online = self.model(next_states)
            next_movement_actions = next_movement_q_online.max(1)[1]
            next_aim_actions = next_aim_q_online.max(1)[1]
            
            next_movement_q_target, next_aim_q_target = self.target_model(next_states)
            next_movement_q = next_movement_q_target.gather(1, next_movement_actions.unsqueeze(1)).squeeze()
            next_aim_q = next_aim_q_target.gather(1, next_aim_actions.unsqueeze(1)).squeeze()
        
        # 改进的目标Q值计算：分别但平衡的奖励分配
        movement_reward_weight = 0.6  # 移动头权重稍高，因为移动头学习更困难
        aim_reward_weight = 0.4
        
        # 根据动作类型智能分配奖励权重
        for i in range(len(rewards)):
            if movement_actions[i] == 0:  # 静止动作，降低移动头权重
                movement_reward_weight_batch = 0.3
                aim_reward_weight_batch = 0.7
            else:  # 有移动，增加移动头权重
                movement_reward_weight_batch = 0.7
                aim_reward_weight_batch = 0.3
            
            target_movement_q = rewards[i] * movement_reward_weight_batch + (self.gamma * next_movement_q[i] * ~dones[i])
            target_aim_q = rewards[i] * aim_reward_weight_batch + (self.gamma * next_aim_q[i] * ~dones[i])
        
        # 重新计算目标Q值张量
        target_movement_q = rewards * movement_reward_weight + (self.gamma * next_movement_q * ~dones)
        target_aim_q = rewards * aim_reward_weight + (self.gamma * next_aim_q * ~dones)
        
        # 分别计算移动和瞄准的损失，使用Huber Loss减少训练不稳定
        movement_loss = nn.HuberLoss()(current_movement_q.squeeze(), target_movement_q)
        aim_loss = nn.HuberLoss()(current_aim_q.squeeze(), target_aim_q)
        
        # 动态损失权重：防止某个头主导训练
        movement_loss_val = movement_loss.item()
        aim_loss_val = aim_loss.item()
        
        # 如果损失差异过大，动态调整权重
        if movement_loss_val > aim_loss_val * 2:
            movement_weight = 0.3
            aim_weight = 0.7
        elif aim_loss_val > movement_loss_val * 2:
            movement_weight = 0.7
            aim_weight = 0.3
        else:
            movement_weight = 0.5
            aim_weight = 0.5
        
        loss = movement_weight * movement_loss + aim_weight * aim_loss
        
        # 应用重要性采样权重
        weighted_loss = (weights * loss).mean()
        
        # 梯度裁剪，防止训练不稳定
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新优先级（基于TD误差）
        with torch.no_grad():
            td_errors = torch.abs(current_movement_q.squeeze() - target_movement_q) + \
                       torch.abs(current_aim_q.squeeze() - target_aim_q)
            priorities = td_errors.cpu().numpy()
            self.memory.update_priorities(indices, priorities)
        
        # 更新目标网络
        self.train_count += 1
        if self.train_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        # 记录损失用于监控
        if not hasattr(self, 'loss_history'):
            self.loss_history = {'movement': [], 'aim': [], 'total': []}
        self.loss_history['movement'].append(movement_loss_val)
        self.loss_history['aim'].append(aim_loss_val)
        self.loss_history['total'].append(weighted_loss.item())
            
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # 智能探索策略：偏向移动，解决移动头停滞
            if not hasattr(self, 'exploration_stats'):
                self.exploration_stats = {'movement': [0]*5, 'aim': [0]*3}
            
            # 偏向移动探索：60%概率探索移动，40%概率探索瞄准
            if random.random() < 0.6:
                # 探索移动动作（优先选择非静止动作）
                movement_probs = [0.1, 0.3, 0.3, 0.15, 0.15]  # 降低静止概率
                movement_action = random.choices(range(5), weights=movement_probs)[0]
                # 瞄准动作使用当前最优策略
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, aim_q = self.model(state_tensor)
                aim_action = aim_q.argmax().item()
            else:
                # 探索瞄准动作
                aim_action = random.randint(0, 2)
                # 移动动作使用当前最优策略
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                movement_q, _ = self.model(state_tensor)
                movement_action = movement_q.argmax().item()
            
            # 记录探索统计
            self.exploration_stats['movement'][movement_action] += 1
            self.exploration_stats['aim'][aim_action] += 1
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            movement_q, aim_q = self.model(state_tensor)
            movement_action = movement_q.argmax().item()
            aim_action = aim_q.argmax().item()
        
        return movement_action, aim_action
    
    def get_combined_action(self, movement_action, aim_action):
        """将双动作转换为游戏可执行的动作列表"""
        actions = []
        if movement_action in MOVEMENT_ACTIONS:
            actions.append(MOVEMENT_ACTIONS[movement_action])
        if aim_action in AIM_ACTIONS:
            actions.append(AIM_ACTIONS[aim_action])
        return actions

def train():
    game = tankgame.TankGame(render=False)
    agent = SimpleAgent()
    scores = []
    game_scores = []  # 游戏内部分数
    recent_losses = []
    
    print("开始训练改进的双动作模型...")
    print("关键改进：")
    print("- 重构奖励函数，统一奖励信号")
    print("- 平衡双头网络训练")
    print("- 智能探索策略")
    print("- 学习率调度和稳定性改进")
    print("- 增强监控指标")
    print()
    
    for episode in range(2000):  # 增加训练轮数
        state = game.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        
        while True:
            movement_action, aim_action = agent.get_action(state, training=True)
            actions = agent.get_combined_action(movement_action, aim_action)
            game.do_actions(actions)
            reward, done = game.step()
            next_state = game.get_state()
            
            agent.remember(state, movement_action, aim_action, reward, next_state, done)
            
            # 每5步训练一次，更频繁的训练
            if steps % 5 == 0:
                agent.train()
                if hasattr(agent, 'loss_history') and agent.loss_history['total']:
                    episode_losses.append(agent.loss_history['total'][-1])
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or steps > 800:  # 增加最大步数
                break
        
        scores.append(total_reward)
        game_scores.append(game.score)
        
        # 学习率调度
        if len(scores) >= 10:
            avg_recent_score = np.mean(scores[-10:])
            agent.scheduler.step(avg_recent_score)
        
        # 每回合衰减探索率
        if agent.epsilon > 0.05:  # 提高最低探索率
            agent.epsilon *= 0.998  # 更慢的衰减
        
        # 详细的训练监控
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            avg_game_score = np.mean(game_scores[-50:]) if len(game_scores) >= 50 else np.mean(game_scores)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            print(f"回合 {episode:4d}: "
                  f"平均奖励={avg_score:6.1f}, "
                  f"平均游戏分数={avg_game_score:6.1f}, "
                  f"当前奖励={total_reward:6.1f}, "
                  f"游戏分数={game.score:3d}, "
                  f"ε={agent.epsilon:.3f}, "
                  f"损失={avg_loss:.4f}")
            
            # 详细监控双头训练情况
            if hasattr(agent, 'loss_history') and len(agent.loss_history['movement']) > 0:
                recent_movement_loss = np.mean(agent.loss_history['movement'][-20:])
                recent_aim_loss = np.mean(agent.loss_history['aim'][-20:])
                print(f"         双头损失 - 移动头: {recent_movement_loss:.4f}, 瞄准头: {recent_aim_loss:.4f}")
            
            # 探索统计
            if hasattr(agent, 'exploration_stats'):
                total_movement = sum(agent.exploration_stats['movement'])
                total_aim = sum(agent.exploration_stats['aim'])
                if total_movement > 0:
                    movement_idle_ratio = agent.exploration_stats['movement'][0] / total_movement
                    print(f"         探索统计 - 静止比例: {movement_idle_ratio:.2f}")
            
            if episode % 200 == 0:
                torch.save(agent.model.state_dict(), f"improved_dual_model_{episode}.pth")
                print(f"         模型已保存: improved_dual_model_{episode}.pth")
    
    torch.save(agent.model.state_dict(), "improved_dual_final_model.pth")
    print(f"\n训练完成！改进的双动作模型已保存为 improved_dual_final_model.pth")
    print(f"最终平均奖励: {np.mean(scores[-100:]):.1f}")
    print(f"最终平均游戏分数: {np.mean(game_scores[-100:]):.1f}")
    
    # 保存训练历史
    if hasattr(agent, 'loss_history'):
        np.savez("training_history.npz", 
                 scores=scores, 
                 game_scores=game_scores,
                 movement_losses=agent.loss_history['movement'],
                 aim_losses=agent.loss_history['aim'])
        print("训练历史已保存到 training_history.npz")

def test():
    game = tankgame.TankGame(render=True)
    agent = SimpleAgent()
    
    try:
        agent.model.load_state_dict(torch.load("dual_final_model.pth"))
        print("加载双动作模型成功")
    except:
        try:
            agent.model.load_state_dict(torch.load("final_model.pth"))
            print("加载旧模型成功，将使用兼容模式")
        except:
            print("未找到训练好的模型，使用随机策略")
    
    agent.epsilon = 0.0
    
    print("开始测试双动作模型...")
    state = game.reset()
    
    while True:
        movement_action, aim_action = agent.get_action(state, training=False)
        actions = agent.get_combined_action(movement_action, aim_action)
        game.do_actions(actions)
        reward, done = game.step()
        state = game.get_state()
        
        if done:
            print(f"游戏结束！最终分数: {game.score}")
            break

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        train()