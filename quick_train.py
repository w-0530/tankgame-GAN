import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
import tankgame

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 512),  # 修复：12维 -> 20维状态向量
            nn.LayerNorm(512),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),  # 增加一层全连接
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )
    
    def forward(self, x):
        return self.net(x)

class SimpleAgent:
    def __init__(self):
        self.model = SimpleNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=50000)
        self.epsilon = 0.5
        self.gamma = 0.99
        self.batch_size = 128
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.BoolTensor([b[4] for b in batch])
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > 0.01:
            self.epsilon *= 0.995
            
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # 更智能的探索策略
            if random.random() < 0.2:  # 20%概率完全随机
                return random.randint(0, 7)
            else:  # 80%概率基于状态的智能探索
                # 检查是否瞄准敌人
                enemy_x, enemy_y = state[3], state[4]
                player_x, player_y = state[0], state[1]
                aim_angle = state[2]
                
                # 计算到敌人的角度
                if enemy_x > 0 and enemy_y > 0:  # 有敌人
                    dx = enemy_x - player_x
                    dy = enemy_y - player_y
                    target_angle = math.atan2(-dy, dx) % (2*math.pi)
                    angle_diff = abs(aim_angle - target_angle)
                    angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                    
                    # 如果瞄准了敌人，优先射击
                    if angle_diff < 0.5 and random.random() < 0.7:  # 约30度内
                        return 7  # 射击
                    # 否则优先调整炮管
                    elif random.random() < 0.8:
                        return random.choice([5, 6])  # 旋转炮管
                    else:
                        return random.choice([1, 2, 3, 4])  # 移动
                else:
                    return random.randint(0, 7)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return q_values.argmax().item()

def quick_train():
    game = tankgame.TankGame(render=False)
    agent = SimpleAgent()
    scores = []
    
    print("开始快速训练...")
    
    for episode in range(50):
        state = game.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(state, training=True)
            game.do_action(action)
            reward, done = game.step()
            next_state = game.get_state()
            
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or steps > 500:  # 限制步数避免卡死
                break
        
        scores.append(total_reward)
        
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"回合 {episode}: 平均分数 = {avg_score:.2f}, 总奖励 = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}")
            
    torch.save(agent.model.state_dict(), "quick_model.pth")
    print("快速训练完成！模型已保存为 quick_model.pth")
    return agent

def quick_test(agent):
    game = tankgame.TankGame(render=True)
    agent.epsilon = 0.0
    
    print("开始快速测试...")
    state = game.reset()
    
    done = False
    for step in range(1000):  # 限制测试步数
        action = agent.get_action(state, training=False)
        game.do_action(action)
        reward, done = game.step()
        state = game.get_state()
        
        if done:
            print(f"游戏结束！最终分数: {game.score}, 步数: {step}")
            break
            
    if not done:
        print(f"测试结束！最终分数: {game.score}")

if __name__ == "__main__":
    agent = quick_train()
    quick_test(agent)