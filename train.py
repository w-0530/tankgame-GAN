#!/usr/bin/env python3
"""
最终优化训练脚本 - 1000回合版本
基于之前测试的最佳改进策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import tankgame
import time
import math

class FinalOptimizedNet(nn.Module):
    """最终优化网络架构"""
    def __init__(self):
        super().__init__()
        
        # 专注核心特征的网络结构
        self.shared_layers = nn.Sequential(
            nn.Linear(67, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 96),
            nn.ReLU()
        )
        
        # 移动头 - 简化但有效
        self.movement_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 5)
        )
        
        # 瞄准头 - 重点优化
        self.aim_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 3)
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        movement_q = self.movement_head(shared_features)
        aim_q = self.aim_head(shared_features)
        return movement_q, aim_q

class FinalOptimizedAgent:
    """最终优化智能体"""
    def __init__(self):
        self.model = FinalOptimizedNet()
        self.target_model = FinalOptimizedNet()
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 优化的优化器设置
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.0005,
            weight_decay=1e-4
        )
        
        # 稳定的经验回放
        self.memory = deque(maxlen=15000)
        
        # 优化的探索策略
        self.epsilon = 0.6
        self.epsilon_decay = 0.997
        self.min_epsilon = 0.02
        
        # 核心训练参数
        self.gamma = 0.98
        self.batch_size = 64
        self.train_count = 0
        
        # 训练统计
        self.loss_history = []
        self.best_score = 0
        
    def remember(self, state, movement_action, aim_action, reward, next_state, done):
        self.memory.append((state, movement_action, aim_action, reward, next_state, done))
    
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # 智能探索策略
            if random.random() < 0.7:
                movement_action = random.choices([0, 1, 2, 3, 4], weights=[0.1, 0.25, 0.25, 0.2, 0.2])[0]
                aim_action = random.choices([0, 1, 2], weights=[0.25, 0.25, 0.5])[0]
            else:
                movement_action = random.randint(0, 4)
                aim_action = random.randint(0, 2)
            return movement_action, aim_action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            movement_q, aim_q = self.model(state_tensor)
            movement_action = movement_q.argmax().item()
            aim_action = aim_q.argmax().item()
        
        return movement_action, aim_action
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([torch.FloatTensor(b[0]) for b in batch])
        movement_actions = torch.LongTensor([b[1] for b in batch])
        aim_actions = torch.LongTensor([b[2] for b in batch])
        rewards = torch.FloatTensor([b[3] for b in batch])
        next_states = torch.stack([torch.FloatTensor(b[4]) for b in batch])
        dones = torch.BoolTensor([b[5] for b in batch])
        
        # 当前Q值
        current_movement_q, current_aim_q = self.model(states)
        current_movement_q = current_movement_q.gather(1, movement_actions.unsqueeze(1))
        current_aim_q = current_aim_q.gather(1, aim_actions.unsqueeze(1))
        
        # Double DQN目标Q值
        with torch.no_grad():
            next_movement_q_online, next_aim_q_online = self.model(next_states)
            next_movement_actions = next_movement_q_online.max(1)[1]
            next_aim_actions = next_aim_q_online.max(1)[1]
            
            next_movement_q_target, next_aim_q_target = self.target_model(next_states)
            next_movement_q = next_movement_q_target.gather(1, next_movement_actions.unsqueeze(1)).squeeze()
            next_aim_q = next_aim_q_target.gather(1, next_aim_actions.unsqueeze(1)).squeeze()
        
        # 目标计算
        target_movement_q = rewards + (self.gamma * next_movement_q * ~dones)
        target_aim_q = rewards + (self.gamma * next_aim_q * ~dones)
        
        # 损失计算 - 重点优化瞄准
        movement_loss = nn.SmoothL1Loss()(current_movement_q.squeeze(), target_movement_q)
        aim_loss = nn.SmoothL1Loss()(current_aim_q.squeeze(), target_aim_q)
        
        total_loss = movement_loss + 1.8 * aim_loss  # 给瞄准更高权重
        
        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.train_count += 1
        
        # 软更新目标网络
        if self.train_count % 5 == 0:
            tau = 0.15
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        return total_loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def calculate_optimized_reward(game, last_score, last_lives, action_info):
    """优化奖励函数 - 平衡且有效"""
    reward = 0
    
    # 1. 核心击杀奖励 - 最重要
    score_delta = game.score - last_score
    if score_delta > 0:
        reward += score_delta * 30  # 强击杀奖励
    
    # 2. 生存奖励
    if game.player.alive:
        reward += 0.5
    
    # 3. 被击中惩罚
    if game.player.lives < last_lives:
        reward -= 20
    
    # 4. 战术位置奖励
    enemy = game._get_nearest_enemy()
    if enemy and enemy.alive:
        dist = tankgame.distance_between(game.player.x, game.player.y, enemy.x, enemy.y)
        
        # 距离奖励
        if 200 <= dist <= 350:
            reward += 2.0
        elif 150 <= dist < 200:
            reward += 1.0
        elif dist < 120:
            reward -= 0.8
        
        # 精确瞄准奖励
        dx = enemy.x - game.player.x
        dy = enemy.y - game.player.y
        target_angle = math.atan2(-dy, dx) % (2 * math.pi)
        angle_diff = abs(game.player.aim_angle - target_angle)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        
        if angle_diff < math.pi/45:  # 4度内
            reward += 4.0
        elif angle_diff < math.pi/22:  # 8度内
            reward += 2.5
        elif angle_diff < math.pi/12:  # 15度内
            reward += 1.2
        
        # 射击奖励 - 仅在瞄准良好时
        if action_info['aim_action'] == 2:  # 射击动作
            if angle_diff < math.pi/18:  # 10度内射击
                reward += 6.0
            elif angle_diff < math.pi/10:  # 18度内
                reward += 3.0
            else:
                reward -= 1.5  # 随意射击惩罚
        
        # 接近敌人但不危险
        if 100 <= dist <= 200:
            reward += 1.2
    
    return reward

def train():
    """最终优化训练 - 1000回合"""
    print("🚀 最终优化训练 (1000回合)")
    print("=" * 60)
    print("核心改进：")
    print("- 优化网络架构 (67→128→96)")
    print("- 强化击杀奖励 (30倍)")
    print("- 智能探索策略")
    print("- 专注瞄准训练")
    print("- Double DQN + 软更新")
    print("- 自适应学习率")
    print("=" * 60)
    
    game = tankgame.TankGame(render=False)
    agent = FinalOptimizedAgent()
    
    scores = []
    game_scores = []
    start_time = time.time()
    
    # 保存检查点的变量
    best_model_score = 0
    last_save_time = time.time()
    
    for episode in range(1000):
        state = game.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        
        last_score = 0
        last_lives = game.player.lives
        
        while True:
            movement_action, aim_action = agent.get_action(state, training=True)
            action_info = {'aim_action': aim_action}
            
            # 执行动作
            actions = []
            if movement_action == 1:
                actions.append(tankgame.ACTION_UP)
            elif movement_action == 2:
                actions.append(tankgame.ACTION_DOWN)
            elif movement_action == 3:
                actions.append(tankgame.ACTION_LEFT)
            elif movement_action == 4:
                actions.append(tankgame.ACTION_RIGHT)
            
            if aim_action == 0:
                actions.append(tankgame.ACTION_GUN_LEFT)
            elif aim_action == 1:
                actions.append(tankgame.ACTION_GUN_RIGHT)
            elif aim_action == 2:
                actions.append(tankgame.ACTION_SHOOT)
            
            game.do_actions(actions)
            reward, done = game.step()
            
            # 计算优化奖励
            optimized_reward = calculate_optimized_reward(game, last_score, last_lives, action_info)
            combined_reward = reward + optimized_reward
            
            last_score = game.score
            last_lives = game.player.lives
            
            next_state = game.get_state()
            
            agent.remember(state, movement_action, aim_action, combined_reward, next_state, done)
            
            # 训练
            if steps % 1 == 0:
                loss = agent.train()
                if loss:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += combined_reward
            steps += 1
            
            if done or steps > 300:
                break
        
        scores.append(total_reward)
        game_scores.append(game.score)
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 定期保存和报告
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            avg_game_score = np.mean(game_scores[-50:]) if len(game_scores) >= 50 else np.mean(game_scores)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            elapsed_time = time.time() - start_time
            
            print(f"回合 {episode:4d}: "
                  f"平均奖励={avg_score:7.1f}, "
                  f"当前奖励={total_reward:7.1f}, "
                  f"平均分={avg_game_score:5.1f}, "
                  f"游戏分={game.score:3d}, "
                  f"ε={agent.epsilon:.3f}, "
                  f"损失={avg_loss:.1f}, "
                  f"用时={elapsed_time/60:.1f}min")
            
            # 保存最佳模型
            if avg_game_score > best_model_score:
                best_model_score = avg_game_score
                torch.save(agent.model.state_dict(), "best_model.pth")
                print(f"          🏆 新最佳模型! 平均分: {best_model_score:.1f}")
            
            # 定期备份
            current_time = time.time()
            if current_time - last_save_time > 300:  # 每5分钟备份一次
                torch.save(agent.model.state_dict(), f"backup_model_ep{episode}.pth")
                last_save_time = current_time
                print(f"          💾 备份模型已保存: backup_model_ep{episode}.pth")
        
        # 更早一些的阶段性报告
        elif episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            avg_game_score = np.mean(game_scores[-10:]) if len(game_scores) >= 10 else np.mean(game_scores)
            
            print(f"回合 {episode:4d}: 平均奖励={avg_score:7.1f}, 平均分={avg_game_score:5.1f}, ε={agent.epsilon:.3f}")
    
    # 最终结果
    total_time = time.time() - start_time
    final_avg_score = np.mean(scores[-50:])
    final_avg_game_score = np.mean(game_scores[-50:])
    
    print(f"\n🏆 最终优化训练完成！")
    print(f"总用时: {total_time/60:.1f}分钟")
    print(f"最终平均奖励: {final_avg_score:.1f}")
    print(f"最终平均游戏分数: {final_avg_game_score:.1f}")
    print(f"最高游戏分数: {max(game_scores):.1f}")
    print(f"训练回合数: {len(scores)}")
    
    # 保存最终模型
    torch.save(agent.model.state_dict(), "final_model_1000.pth")
    print(f"最终模型已保存为: final_model_1000.pth")
    print(f"最佳模型保存为: best_model.pth (平均分: {best_model_score:.1f})")
    
    return scores, game_scores

if __name__ == "__main__":
    scores, game_scores = train()