#!/usr/bin/env python3
"""
测试可视化脚本 - 实时显示AI测试过程和游戏状态
包含游戏画面、AI决策信息、性能指标等可视化功能
"""
import torch
import torch.nn as nn
import numpy as np
import tankgame
import time
import pygame
import math
from collections import deque

# 可视化窗口设置
VIS_WIDTH = 1600
VIS_HEIGHT = 800
GAME_WIDTH = 1200
GAME_HEIGHT = 600
INFO_WIDTH = 400

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GREEN = (0, 150, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

class FinalOptimizedNet(nn.Module):
    """最终优化网络架构"""
    def __init__(self):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(67, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 96),
            nn.ReLU()
        )
        
        self.movement_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 5)
        )
        
        self.aim_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 3)
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        movement_q = self.movement_head(shared_features)
        aim_q = self.aim_head(shared_features)
        return movement_q, aim_q

class TestVisualizer:
    """测试可视化器"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((VIS_WIDTH, VIS_HEIGHT))
        pygame.display.set_caption("坦克AI测试可视化")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont(None, 24)
        self.font_medium = pygame.font.SysFont(None, 32)
        self.font_large = pygame.font.SysFont(None, 48)
        
        # 游戏实例
        self.game = tankgame.TankGame(render=False)
        
        # 性能统计
        self.episode_scores = deque(maxlen=50)
        self.episode_game_scores = deque(maxlen=50)
        self.current_episode = 0
        self.total_episodes = 20
        self.test_start_time = time.time()
        
        # AI决策信息
        self.last_movement_q = None
        self.last_aim_q = None
        self.last_movement_action = None
        self.last_aim_action = None
        
        # 图表数据
        self.score_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
    def load_model(self, model_path):
        """加载模型"""
        self.model = FinalOptimizedNet()
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"✓ 成功加载模型: {model_path}")
            return True
        except FileNotFoundError:
            print(f"✗ 模型文件不存在: {model_path}")
            return False
    
    def get_ai_decision(self, state):
        """获取AI决策信息"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            movement_q, aim_q = self.model(state_tensor)
            movement_action = movement_q.argmax().item()
            aim_action = aim_q.argmax().item()
            
            # 保存决策信息
            self.last_movement_q = movement_q.squeeze().cpu().numpy()
            self.last_aim_q = aim_q.squeeze().cpu().numpy()
            self.last_movement_action = movement_action
            self.last_aim_action = aim_action
            
            return movement_action, aim_action
    
    def draw_game_area(self):
        """绘制游戏区域"""
        # 游戏背景
        game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, game_rect)
        pygame.draw.rect(self.screen, WHITE, game_rect, 2)
        
        # 绘制游戏对象
        if not self.game.game_over:
            # 玩家坦克
            player = self.game.player
            if player.alive:
                pygame.draw.rect(self.screen, GREEN, 
                               (player.x - player.size//2, player.y - player.size//2, 
                                player.size, player.size))
                # 炮管
                gun_x = player.x + math.cos(player.aim_angle) * 25
                gun_y = player.y - math.sin(player.aim_angle) * 25
                pygame.draw.line(self.screen, WHITE, (player.x, player.y), (gun_x, gun_y), 3)
                
                # 瞄准线
                aim_x = player.x + math.cos(player.aim_angle) * 100
                aim_y = player.y - math.sin(player.aim_angle) * 100
                pygame.draw.line(self.screen, YELLOW, (player.x, player.y), (aim_x, aim_y), 1)
            
            # 敌人坦克
            for enemy in self.game.enemies:
                if enemy.alive:
                    pygame.draw.rect(self.screen, RED,
                                   (enemy.x - enemy.size//2, enemy.y - enemy.size//2,
                                    enemy.size, enemy.size))
                    # 敌人炮管
                    gun_x = enemy.x + math.cos(enemy.aim_angle) * 25
                    gun_y = enemy.y - math.sin(enemy.aim_angle) * 25
                    pygame.draw.line(self.screen, WHITE, (enemy.x, enemy.y), (gun_x, gun_y), 3)
            
            # 子弹
            for bullet in self.game.bullets:
                color = YELLOW if bullet.is_player_bullet else RED
                pygame.draw.circle(self.screen, color, (int(bullet.x), int(bullet.y)), bullet.radius)
        
        # 游戏信息
        info_y = 10
        score_text = self.font_medium.render(f"Score: {self.game.score}", True, WHITE)
        self.screen.blit(score_text, (10, info_y))
        
        lives_text = self.font_medium.render(f"Lives: {self.game.player.lives}", True, RED)
        self.screen.blit(lives_text, (200, info_y))
        
        time_text = self.font_medium.render(f"Time: {self.game.remaining_time}s", True, WHITE)
        self.screen.blit(time_text, (350, info_y))
        
        episode_text = self.font_medium.render(f"Episode: {self.current_episode}/{self.total_episodes}", True, WHITE)
        self.screen.blit(episode_text, (500, info_y))
    
    def draw_ai_info(self):
        """绘制AI决策信息"""
        info_rect = pygame.Rect(GAME_WIDTH, 0, INFO_WIDTH, VIS_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GREEN, info_rect)
        pygame.draw.rect(self.screen, WHITE, info_rect, 2)
        
        y_offset = 20
        
        # 标题
        title_text = self.font_large.render("AI决策信息", True, WHITE)
        self.screen.blit(title_text, (GAME_WIDTH + 50, y_offset))
        y_offset += 60
        
        # 动作信息
        if self.last_movement_action is not None:
            movement_names = ["静止", "上", "下", "左", "右"]
            aim_names = ["炮管左", "炮管右", "射击"]
            
            move_text = self.font_medium.render(f"移动: {movement_names[self.last_movement_action]}", True, WHITE)
            self.screen.blit(move_text, (GAME_WIDTH + 20, y_offset))
            y_offset += 35
            
            aim_text = self.font_medium.render(f"瞄准: {aim_names[self.last_aim_action]}", True, WHITE)
            self.screen.blit(aim_text, (GAME_WIDTH + 20, y_offset))
            y_offset += 50
        
        # Q值信息
        if self.last_movement_q is not None:
            q_title = self.font_medium.render("移动Q值:", True, YELLOW)
            self.screen.blit(q_title, (GAME_WIDTH + 20, y_offset))
            y_offset += 30
            
            for i, q in enumerate(self.last_movement_q):
                action_name = ["静", "上", "下", "左", "右"][i]
                q_text = self.font_small.render(f"{action_name}: {q:.2f}", True, WHITE)
                self.screen.blit(q_text, (GAME_WIDTH + 20, y_offset))
                y_offset += 25
            
            y_offset += 10
            
            q_title = self.font_medium.render("瞄准Q值:", True, YELLOW)
            self.screen.blit(q_title, (GAME_WIDTH + 20, y_offset))
            y_offset += 30
            
            for i, q in enumerate(self.last_aim_q):
                action_name = ["左转", "右转", "射击"][i]
                q_text = self.font_small.render(f"{action_name}: {q:.2f}", True, WHITE)
                self.screen.blit(q_text, (GAME_WIDTH + 20, y_offset))
                y_offset += 25
            
            y_offset += 20
        
        # 性能统计
        stats_title = self.font_medium.render("性能统计:", True, YELLOW)
        self.screen.blit(stats_title, (GAME_WIDTH + 20, y_offset))
        y_offset += 30
        
        if self.episode_scores:
            avg_score = np.mean(self.episode_scores)
            avg_game_score = np.mean(self.episode_game_scores)
            
            score_text = self.font_small.render(f"平均奖励: {avg_score:.1f}", True, WHITE)
            self.screen.blit(score_text, (GAME_WIDTH + 20, y_offset))
            y_offset += 25
            
            game_score_text = self.font_small.render(f"平均分数: {avg_game_score:.1f}", True, WHITE)
            self.screen.blit(game_score_text, (GAME_WIDTH + 20, y_offset))
            y_offset += 25
            
            if self.episode_game_scores:
                max_score = max(self.episode_game_scores)
                max_text = self.font_small.render(f"最高分数: {max_score}", True, WHITE)
                self.screen.blit(max_text, (GAME_WIDTH + 20, y_offset))
                y_offset += 25
                
                kills_text = self.font_small.render(f"总击杀: {int(sum(self.episode_game_scores)/70)}", True, WHITE)
                self.screen.blit(kills_text, (GAME_WIDTH + 20, y_offset))
                y_offset += 25
        
        # 时间信息
        elapsed_time = time.time() - self.test_start_time
        time_text = self.font_small.render(f"测试时间: {elapsed_time:.1f}s", True, WHITE)
        self.screen.blit(time_text, (GAME_WIDTH + 20, y_offset))
    
    def draw_mini_chart(self):
        """绘制迷你图表"""
        if len(self.score_history) < 2:
            return
        
        chart_rect = pygame.Rect(GAME_WIDTH + 20, VIS_HEIGHT - 200, INFO_WIDTH - 40, 180)
        pygame.draw.rect(self.screen, BLACK, chart_rect)
        pygame.draw.rect(self.screen, WHITE, chart_rect, 1)
        
        # 标题
        chart_title = self.font_small.render("分数历史", True, WHITE)
        self.screen.blit(chart_title, (chart_rect.x + 10, chart_rect.y - 20))
        
        # 绘制分数曲线
        scores = list(self.score_history)
        if scores:
            max_score = max(scores) if max(scores) > 0 else 1
            min_score = min(scores)
            score_range = max_score - min_score if max_score != min_score else 1
            
            points = []
            for i, score in enumerate(scores):
                x = chart_rect.x + 10 + (i * (chart_rect.width - 20) // len(scores))
                y = chart_rect.y + chart_rect.height - 10 - int((score - min_score) / score_range * (chart_rect.height - 20))
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, GREEN, False, points, 2)
    
    def run_test_episode(self):
        """运行单个测试回合"""
        state = self.game.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 获取AI决策
            movement_action, aim_action = self.get_ai_decision(state)
            
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
            
            self.game.do_actions(actions)
            reward, done = self.game.step()
            next_state = self.game.get_state()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 更新历史
            self.score_history.append(self.game.score)
            self.reward_history.append(total_reward)
            
            if done or steps > 300:
                break
        
        # 记录结果
        self.episode_scores.append(total_reward)
        self.episode_game_scores.append(self.game.score)
        
        return total_reward, self.game.score
    
    def run_visualized_test(self, model_path, episodes=20):
        """运行可视化测试"""
        if not self.load_model(model_path):
            return
        
        self.total_episodes = episodes
        self.current_episode = 0
        self.test_start_time = time.time()
        
        print(f"🎯 开始可视化测试: {model_path}")
        print(f"测试回合数: {episodes}")
        
        running = True
        episode_complete = False
        episode_delay = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        episode_complete = True
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # 运行测试回合
            if self.current_episode < self.total_episodes and not episode_complete:
                if episode_delay > 0:
                    episode_delay -= 1
                else:
                    # 运行一个回合
                    total_reward, game_score = self.run_test_episode()
                    self.current_episode += 1
                    episode_complete = True
                    episode_delay = 60  # 显示1秒结果
                    
                    print(f"回合 {self.current_episode}: 奖励={total_reward:.1f}, 分数={game_score}")
            
            elif episode_complete and episode_delay > 0:
                episode_delay -= 1
                if episode_delay == 0:
                    episode_complete = False
                    # 重置游戏准备下一回合
                    self.game.reset()
            
            # 绘制界面
            self.screen.fill(BLACK)
            self.draw_game_area()
            self.draw_ai_info()
            self.draw_mini_chart()
            
            # 如果回合完成，显示结果
            if episode_complete and self.episode_game_scores:
                result_text = self.font_large.render(f"回合 {self.current_episode} 完成!", True, YELLOW)
                text_rect = result_text.get_rect(center=(GAME_WIDTH//2, GAME_HEIGHT//2))
                pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 10))
                self.screen.blit(result_text, text_rect)
                
                score_text = self.font_medium.render(f"分数: {self.episode_game_scores[-1]}", True, WHITE)
                score_rect = score_text.get_rect(center=(GAME_WIDTH//2, GAME_HEIGHT//2 + 50))
                self.screen.blit(score_text, score_rect)
            
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        # 最终统计
        if self.episode_scores:
            avg_score = np.mean(self.episode_scores)
            avg_game_score = np.mean(self.episode_game_scores)
            
            print(f"\n📊 测试完成!")
            print(f"平均奖励: {avg_score:.1f} ± {np.std(self.episode_scores):.1f}")
            print(f"平均分数: {avg_game_score:.1f} ± {np.std(self.episode_game_scores):.1f}")
            print(f"最高分数: {max(self.episode_game_scores)}")
            print(f"总击杀数: {int(sum(self.episode_game_scores)/70)}")
        
        pygame.quit()

def main():
    """主函数"""
    print("🎯 坦克AI测试可视化")
    print("=" * 40)
    
    # 检查可用的模型
    models_to_test = []
    for model_path in ["best_model.pth", "final_model_1000.pth"]:
        try:
            torch.load(model_path, map_location='cpu')
            models_to_test.append(model_path)
        except FileNotFoundError:
            pass
    
    if not models_to_test:
        print("❌ 未找到可用的模型文件")
        return
    
    print(f"找到模型: {', '.join(models_to_test)}")
    
    # 创建可视化器
    visualizer = TestVisualizer()
    
    # 运行测试
    visualizer.run_visualized_test(models_to_test[0], episodes=20)

if __name__ == "__main__":
    main()