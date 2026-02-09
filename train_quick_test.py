import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import tankgame

# 简化的鲁棒性网络
class CompactRobustNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(33, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        self.movement_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 5)
        )
        
        self.aim_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3)
        )
        
        # 保守初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=0.3)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        movement_q = self.movement_head(shared_features)
        aim_q = self.aim_head(shared_features)
        return movement_q, aim_q

class CompactRobustAgent:
    def __init__(self):
        self.model = CompactRobustNet()
        self.target_model = CompactRobustNet()
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 更保守的优化器设置
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=0.02)
        
        # 简单经验回放
        self.memory = []
        self.memory_capacity = 10000
        
        # 自适应探索
        self.epsilon = 0.3
        self.min_epsilon = 0.02
        self.performance_history = []
        
        # 训练参数
        self.gamma = 0.99
        self.batch_size = 128
        self.update_target_freq = 1000
        self.train_count = 0
        
        # 稳定性参数
        self.max_grad_norm = 0.3
        self.loss_history = []
        
        # 损失权重
        self.movement_weight = 0.6
        self.aim_weight = 0.4
    
    def remember(self, state, movement_action, aim_action, reward, next_state, done):
        experience = (state, movement_action, aim_action, reward, next_state, done)
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)  # 移除最旧的经验
        self.memory.append(experience)
    
    def adaptive_epsilon(self, reward):
        """基于表现自适应调整探索率"""
        self.performance_history.append(reward)
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        if len(self.performance_history) >= 5:
            recent_avg = np.mean(self.performance_history[-3:])
            older_avg = np.mean(self.performance_history[-5:-2]) if len(self.performance_history) >= 5 else recent_avg
            
            # 表现下降则增加探索
            if recent_avg < older_avg * 0.8:
                self.epsilon = min(0.4, self.epsilon * 1.05)
            # 表现提升则减少探索
            elif recent_avg > older_avg * 1.1:
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.98)
        
        return self.epsilon
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([b[0] for b in batch])
        movement_actions = torch.LongTensor([b[1] for b in batch])
        aim_actions = torch.LongTensor([b[2] for b in batch])
        rewards = torch.FloatTensor([b[3] for b in batch])
        next_states = torch.FloatTensor([b[4] for b in batch])
        dones = torch.BoolTensor([b[5] for b in batch])
        
        # 当前Q值
        current_movement_q, current_aim_q = self.model(states)
        current_movement_q = current_movement_q.gather(1, movement_actions.unsqueeze(1))
        current_aim_q = current_aim_q.gather(1, aim_actions.unsqueeze(1))
        
        # 目标Q值（Double DQN）
        with torch.no_grad():
            next_movement_q_online, next_aim_q_online = self.model(next_states)
            next_movement_actions = next_movement_q_online.max(1)[1]
            next_aim_actions = next_aim_q_online.max(1)[1]
            
            next_movement_q_target, next_aim_q_target = self.target_model(next_states)
            next_movement_q = next_movement_q_target.gather(1, next_movement_actions.unsqueeze(1)).squeeze()
            next_aim_q = next_aim_q_target.gather(1, next_aim_actions.unsqueeze(1)).squeeze()
        
        # 目标计算
        target_movement_q = rewards * self.movement_weight + (self.gamma * next_movement_q * ~dones)
        target_aim_q = rewards * self.aim_weight + (self.gamma * next_aim_q * ~dones)
        
        # Huber Loss
        movement_loss = nn.SmoothL1Loss()(current_movement_q.squeeze(), target_movement_q)
        aim_loss = nn.SmoothL1Loss()(current_aim_q.squeeze(), target_aim_q)
        
        # 动态权重调整
        movement_loss_val = movement_loss.item()
        aim_loss_val = aim_loss.item()
        
        if movement_loss_val > aim_loss_val * 1.3:
            self.movement_weight = max(0.4, self.movement_weight - 0.01)
            self.aim_weight = min(0.6, self.aim_weight + 0.01)
        elif aim_loss_val > movement_loss_val * 1.3:
            self.movement_weight = min(0.7, self.movement_weight + 0.01)
            self.aim_weight = max(0.3, self.aim_weight - 0.01)
        
        total_loss = self.movement_weight * movement_loss + self.aim_weight * aim_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 严格梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        
        # 更新目标网络
        self.train_count += 1
        if self.train_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.loss_history.append(total_loss.item())
        return total_loss.item()
    
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # 智能探索：降低静止概率
            if random.random() < 0.8:  # 80%概率探索移动
                movement_probs = [0.02, 0.4, 0.4, 0.09, 0.09]  # 静止仅2%
                movement_action = random.choices(range(5), weights=movement_probs)[0]
                
                # 瞄准用贪心
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, aim_q = self.model(state_tensor)
                aim_action = aim_q.argmax().item()
            else:
                aim_action = random.randint(0, 2)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                movement_q, _ = self.model(state_tensor)
                movement_action = movement_q.argmax().item()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            movement_q, aim_q = self.model(state_tensor)
            movement_action = movement_q.argmax().item()
            aim_action = aim_q.argmax().item()
        
        return movement_action, aim_action
    
    def get_combined_action(self, movement_action, aim_action):
        actions = []
        if movement_action in MOVEMENT_ACTIONS:
            actions.append(MOVEMENT_ACTIONS[movement_action])
        if aim_action in AIM_ACTIONS:
            actions.append(AIM_ACTIONS[aim_action])
        return actions

# 动作映射
MOVEMENT_ACTIONS = {
    0: tankgame.ACTION_IDLE,
    1: tankgame.ACTION_UP,
    2: tankgame.ACTION_DOWN,
    3: tankgame.ACTION_LEFT,
    4: tankgame.ACTION_RIGHT
}

AIM_ACTIONS = {
    0: tankgame.ACTION_GUN_LEFT,
    1: tankgame.ACTION_GUN_RIGHT,
    2: tankgame.ACTION_SHOOT
}

def quick_train():
    game = tankgame.TankGame(render=False)
    agent = CompactRobustAgent()
    scores = []
    game_scores = []
    
    print("开始快速测试鲁棒性改进...")
    print("关键改进：")
    print("- 保守网络架构 + LayerNorm")
    print("- 严格梯度裁剪 (0.3)")
    print("- 自适应探索策略")
    print("- 动态损失权重")
    print("- 智能探索偏向")
    print()
    
    for episode in range(200):  # 快速测试200回合
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
            
            # 每2步训练一次
            if steps % 2 == 0 and len(agent.memory) >= agent.batch_size:
                loss = agent.train()
                if loss:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or steps > 400:
                break
        
        scores.append(total_reward)
        game_scores.append(game.score)
        
        # 自适应探索
        agent.adaptive_epsilon(total_reward)
        
        # 每20回合报告
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            avg_game_score = np.mean(game_scores[-20:]) if len(game_scores) >= 20 else np.mean(game_scores)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            print(f"回合 {episode:3d}: "
                  f"平均奖励={avg_score:7.1f}, "
                  f"平均游戏分数={avg_game_score:6.1f}, "
                  f"当前奖励={total_reward:7.1f}, "
                  f"游戏分数={game.score:3d}, "
                  f"ε={agent.epsilon:.3f}, "
                  f"损失={avg_loss:.4f}")
            
            print(f"         损失权重 - 移动: {agent.movement_weight:.3f}, 瞄准: {agent.aim_weight:.3f}")
            print(f"         经验缓冲: {len(agent.memory)}/{agent.memory_capacity}")
    
    torch.save(agent.model.state_dict(), "compact_robust_model.pth")
    print(f"\n快速测试完成！")
    print(f"最终平均奖励: {np.mean(scores[-20:]):.1f}")
    print(f"最终平均游戏分数: {np.mean(game_scores[-20:]):.1f}")
    print(f"模型已保存为: compact_robust_model.pth")
    
    return scores, game_scores

if __name__ == "__main__":
    quick_train()