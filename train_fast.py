import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
import tankgame
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# 导入游戏常量用于反归一化
SCREEN_WIDTH = tankgame.SCREEN_WIDTH
SCREEN_HEIGHT = tankgame.SCREEN_HEIGHT

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(20, 256),  # 减少网络大小
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 移动头：5个动作（静止、上、下、左、右）
        self.movement_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        # 瞄准头：3个动作（左转、右转、射击）
        self.aim_head = nn.Sequential(
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
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=0.9)
        self.memory = deque(maxlen=50000)
        self.epsilon = 0.5
        self.gamma = 0.99
        self.batch_size = 128  # 减少批次大小提高稳定性，加快学习
        self.update_target_freq = 1000
        self.train_count = 0
    
    def remember(self, state, movement_action, aim_action, reward, next_state, done):
        self.memory.append((state, movement_action, aim_action, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([b[0] for b in batch])
        movement_actions = torch.LongTensor([b[1] for b in batch])
        aim_actions = torch.LongTensor([b[2] for b in batch])
        rewards = torch.FloatTensor([b[3] for b in batch])
        next_states = torch.FloatTensor([b[4] for b in batch])
        dones = torch.BoolTensor([b[5] for b in batch])
        
        # 获取当前Q值
        current_movement_q, current_aim_q = self.model(states)
        current_movement_q = current_movement_q.gather(1, movement_actions.unsqueeze(1))
        current_aim_q = current_aim_q.gather(1, aim_actions.unsqueeze(1))
        
        # 使用Double DQN减少过估计偏差
        with torch.no_grad():
            next_movement_q_online, next_aim_q_online = self.model(next_states)
            next_movement_actions = next_movement_q_online.argmax(dim=1)
            next_aim_actions = next_aim_q_online.argmax(dim=1)
            
            next_movement_q_target, next_aim_q_target = self.target_model(next_states)
            next_movement_q = next_movement_q_target.gather(1, next_movement_actions.unsqueeze(1)).squeeze()
            next_aim_q = next_aim_q_target.gather(1, next_aim_actions.unsqueeze(1)).squeeze()
        
        # 核心修复：分别计算移动和瞄准的奖励贡献
        movement_reward_factor = 0.2  # 移动动作对总奖励的贡献权重
        aim_reward_factor = 0.8     # 瞄准动作对总奖励的贡献权重
        
        # 为瞄准动作添加基础奖励，解决Q值负值问题
        aim_base_reward = 0.1  # 基础瞄准奖励
        
        # 分别计算移动和瞄准的目标Q值
        target_movement_q = rewards * movement_reward_factor + (self.gamma * next_movement_q * ~dones)
        target_aim_q = (rewards * aim_reward_factor + aim_base_reward) + (self.gamma * next_aim_q * ~dones)
        
        # 使用Huber损失减少异常值影响
        movement_loss = nn.SmoothL1Loss()(current_movement_q.squeeze(), target_movement_q)
        aim_loss = nn.SmoothL1Loss()(current_aim_q.squeeze(), target_aim_q)
        loss = movement_loss + aim_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新目标网络
        self.train_count += 1
        if self.train_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # 简化的探索策略：完全随机，让算法自己学习
            movement_action = random.randint(0, 4)
            aim_action = random.randint(0, 2)
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

def run_episode(agent, episode_id):
    """运行单个回合，用于并行训练"""
    game = tankgame.TankGame(render=False)
    state = game.reset()
    total_reward = 0
    steps = 0
    episode_memory = []
    
    while True:
        movement_action, aim_action = agent.get_action(state, training=True)
        actions = agent.get_combined_action(movement_action, aim_action)
        game.do_actions(actions)
        reward, done = game.step()
        next_state = game.get_state()
        
        episode_memory.append((state, movement_action, aim_action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done or steps > 500:
            break
    
    return episode_memory, total_reward

def train_parallel():
    """并行训练版本"""
    agent = SimpleAgent()
    scores = []
    
    print("开始并行训练双动作模型...")
    
    for episode in range(1000):
        # 运行单个回合
        episode_memory, total_reward = run_episode(agent, episode)
        
        # 将回合经验添加到记忆库
        for exp in episode_memory:
            agent.remember(*exp)
        
        # 训练
        agent.train()
        
        # 更新学习率调度器（放在optimizer.step之后）
        if episode > 0:  # 避免第一步就调用
            agent.scheduler.step()
        
        scores.append(total_reward)
        
        # 优化epsilon衰减策略
        if agent.epsilon > 0.1:
            agent.epsilon *= 0.995  # 更快的衰减，减少随机探索
        
        if episode % 50 == 0:  # 增加监控频率
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            
            # 获取当前学习率
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            # 估算Q值分布
            avg_movement_q, avg_aim_q = 0.0, 0.0
            if len(agent.memory) >= 100:
                sample_states = torch.FloatTensor([random.choice(agent.memory)[0] for _ in range(100)])
                with torch.no_grad():
                    movement_q, aim_q = agent.model(sample_states)
                    avg_movement_q = movement_q.mean().item()
                    avg_aim_q = aim_q.mean().item()
                    
            print(f"回合 {episode}: 平均分数 = {avg_score:.1f}, 总奖励 = {total_reward:.1f}, epsilon = {agent.epsilon:.3f}, lr = {current_lr:.6f}")
            if len(agent.memory) >= 100:
                print(f"  Q值分布 - 移动: {avg_movement_q:.3f}, 瞄准: {avg_aim_q:.3f}")
            
            if episode % 200 == 0:  # 更频繁保存模型
                torch.save(agent.model.state_dict(), f"dual_model_{episode}.pth")
    
    torch.save(agent.model.state_dict(), "dual_final_model.pth")
    print("训练完成！双动作模型已保存为 dual_final_model.pth")

def train():
    """单线程训练版本"""
    game = tankgame.TankGame(render=False)
    agent = SimpleAgent()
    scores = []
    
    print("开始训练双动作模型...")
    
    for episode in range(1000):
        state = game.reset()
        total_reward = 0
        steps = 0
        
        while True:
            movement_action, aim_action = agent.get_action(state, training=True)
            actions = agent.get_combined_action(movement_action, aim_action)
            game.do_actions(actions)
            reward, done = game.step()
            next_state = game.get_state()
            
            agent.remember(state, movement_action, aim_action, reward, next_state, done)
            
            # 每步都训练，提高学习效率
            agent.train()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or steps > 500:  # 限制最大步数
                break
        
        scores.append(total_reward)
        
        # 每回合衰减探索率，而不是每步
        if agent.epsilon > 0.05:
            agent.epsilon *= 0.998  # 更缓慢的衰减
        
        if episode % 50 == 0:  # 增加监控频率
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            
            # 获取当前学习率
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            # 估算Q值分布
            avg_movement_q, avg_aim_q = 0.0, 0.0
            if len(agent.memory) >= 100:
                sample_states = torch.FloatTensor([random.choice(agent.memory)[0] for _ in range(100)])
                with torch.no_grad():
                    movement_q, aim_q = agent.model(sample_states)
                    avg_movement_q = movement_q.mean().item()
                    avg_aim_q = aim_q.mean().item()
                    
            print(f"回合 {episode}: 平均分数 = {avg_score:.1f}, 总奖励 = {total_reward:.1f}, epsilon = {agent.epsilon:.3f}, lr = {current_lr:.6f}")
            if len(agent.memory) >= 100:
                print(f"  Q值分布 - 移动: {avg_movement_q:.3f}, 瞄准: {avg_aim_q:.3f}")
            
            if episode % 200 == 0:  # 更频繁保存模型
                torch.save(agent.model.state_dict(), f"dual_model_{episode}.pth")
    
    torch.save(agent.model.state_dict(), "dual_final_model.pth")
    print("训练完成！双动作模型已保存为 dual_final_model.pth")

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
    elif len(sys.argv) > 1 and sys.argv[1] == "parallel":
        train_parallel()
    else:
        train()