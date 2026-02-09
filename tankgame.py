import pygame
import random
import math
import numpy as np

# ====================== 游戏常量（可独立修改，不影响AI） =======================
pygame.init()
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
FPS = 60
TANK_SIZE = 40
BULLET_RADIUS = 5
TANK_SPEED = 4
BULLET_BASE_SPEED = 12
MAX_BOUNCES = 10
ENEMY_BULLET_ANGLE_OFFSET = math.pi/60  # 射击轻微散布，不影响瞄准
ENEMY_MAX_BULLETS = 5
GAME_TIME_LIMIT = 60  # 新增：游戏时间限制（秒），可自定义修改

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
BUTTON_COLOR = (200, 100, 0)
BUTTON_HOVER = (255, 150, 0)
RED_ALERT = (255, 80, 80)  # 新增：时间不足的警示颜色

# 动作常量（和AI统一，0-7，移除未使用的ACTION_MOVE_AIM，避免AI维度不匹配）
ACTION_IDLE = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_GUN_LEFT = 5
ACTION_GUN_RIGHT = 6
ACTION_SHOOT = 7

# ====================== 工具函数（游戏内部使用） =======================
def deg2rad(deg):
    return deg * math.pi / 180.0
def rad2deg(rad):
    return rad * 180.0 / math.pi
def normalize_vector(dx, dy):
    length = math.hypot(dx, dy)
    return (dx/length, dy/length) if length != 0 else (0, 0)
def distance_between(x1, y1, x2, y2):
    return math.hypot(x2-x1, y2-y1)
def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val + 1e-8)
def get_screen_diag():
    return math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)

# ====================== 游戏基础类（纯游戏逻辑） =======================
class Bullet:
    def __init__(self, x, y, angle, is_player_bullet):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = BULLET_BASE_SPEED
        self.is_player_bullet = is_player_bullet
        self.radius = BULLET_RADIUS
        self.bounce_count = 0
        self.max_bounces = MAX_BOUNCES
        self.active = True

    def move(self):
        """移除墙壁碰撞检测，只保留屏幕边界反弹"""
        if not self.active: return
        next_x = self.x + math.cos(self.angle) * self.speed
        next_y = self.y - math.sin(self.angle) * self.speed
        
        # 移除墙壁碰撞检测
        if next_x <= self.radius or next_x >= SCREEN_WIDTH - self.radius:
            self.angle = -self.angle
            self.bounce_count += 1
            next_x = max(self.radius, min(SCREEN_WIDTH - self.radius, next_x))
        if next_y <= self.radius or next_y >= SCREEN_HEIGHT - self.radius:
            self.angle = math.pi - self.angle
            self.bounce_count += 1
            next_y = max(self.radius, min(SCREEN_HEIGHT - self.radius, next_y))
        
        self.x = next_x
        self.y = next_y
        if self.bounce_count >= self.max_bounces:
            self.active = False

    def draw(self, screen):
        if self.active:
            color = YELLOW if self.is_player_bullet else RED
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)

class Tank:
    def __init__(self, x, y, color, is_player=False):
        self.x = x
        self.y = y
        self.color = color
        self.is_player = is_player
        self.size = TANK_SIZE
        self.speed = TANK_SPEED if is_player else 2
        self.lives = 5 if is_player else 1
        self.cooldown = 0
        self.cooldown_max = 25 if is_player else 40
        self.aim_angle = 0.0  # 角度完全适配pygame子弹发射逻辑
        self.alive = True
        # ============== 核心修改1：新增AI自动开火标记 ==============
        self.auto_shoot = False  # AI控制时生效，手动游玩不影响

    def draw(self, screen):
        if not self.alive: return
        rect = pygame.Rect(self.x-self.size//2, self.y-self.size//2, self.size, self.size)
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        # 炮管绘制和子弹发射用完全相同的角度计算，保证朝向和子弹一致
        gun_x = self.x + math.cos(self.aim_angle) * 25
        gun_y = self.y - math.sin(self.aim_angle) * 25
        pygame.draw.line(screen, BLACK, (self.x, self.y), (gun_x, gun_y), 5)

    def move(self, dx, dy):
        """移除墙壁碰撞检测，只保留屏幕边界检测"""
        if not self.alive: return
        # 处理对角线移动，使移动速度一致
        if dx != 0 and dy != 0:
            dx *= 0.7071  # 1/√2 ≈ 0.7071，保持对角线移动速度与单方向一致
            dy *= 0.7071
            
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        # 只检查屏幕边界，移除墙壁碰撞检测
        new_x = max(self.size//2, min(SCREEN_WIDTH - self.size//2, new_x))
        new_y = max(self.size//2, min(SCREEN_HEIGHT - self.size//2, new_y))
        
        self.x = new_x
        self.y = new_y

    def rotate_gun(self, d_angle):
        self.aim_angle += d_angle
        self.aim_angle %= (2 * math.pi)

    def shoot(self):
        if self.cooldown > 0 or not self.alive: return None
        self.cooldown = self.cooldown_max
        # 子弹发射位置和炮管朝向完全一致
        bullet_x = self.x + math.cos(self.aim_angle) * (self.size//2 + BULLET_RADIUS)
        bullet_y = self.y - math.sin(self.aim_angle) * (self.size//2 + BULLET_RADIUS)
        return Bullet(bullet_x, bullet_y, self.aim_angle, self.is_player)

    def update_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1
    
    def set_auto_shoot(self, enabled):
        """启用或禁用自动开火功能"""
        self.auto_shoot = enabled

# ====================== 核心游戏类（暴露AI接口，可独立运行） =======================
class TankGame:
    def __init__(self, render=True):
        # 修复点1：重命名布尔属性，避免和render方法重名（核心解决bool不可调用错误）
        self.is_render = render
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) if self.is_render else None
        self.clock = pygame.time.Clock()
        self.small_font = pygame.font.SysFont(None, 36) if self.is_render else None
        self.big_font = pygame.font.SysFont(None, 80) if self.is_render else None
        self.button_font = pygame.font.SysFont(None, 48) if self.is_render else None
        if self.is_render:
            pygame.display.set_caption("坦克大战 - 无障碍物版")
        self.restart_btn = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 50, 200, 60)
        self.game_over = False
        self.reset()

    def reset(self):
        """重置游戏，AI调用：初始化所有游戏对象（含时间计时）"""
        self.player = Tank(SCREEN_WIDTH//2, SCREEN_HEIGHT-TANK_SIZE-20, GREEN, is_player=True)
        self.enemies = [Tank(random.randint(100,700), random.randint(100,200), BLUE) for _ in range(3)]
        self.walls = []  # 移除墙壁
        self.bullets = []
        self.score = 0
        self.step_count = 0
        self.max_steps = 1000
        self.last_lives = self.player.lives
        self.last_score = 0
        self.game_over = False
        # 新增：重置时间计时相关变量
        self.total_frames = 0  # 总帧数（用于计算时间，FPS固定60）
        self.remaining_time = GAME_TIME_LIMIT  # 剩余时间（秒）
        return self.get_state()

    def _get_nearest_enemy(self):
        if not [e for e in self.enemies if e.alive]: return None
        enemies_alive = [e for e in self.enemies if e.alive]
        distances = [distance_between(self.player.x, self.player.y, e.x, e.y) for e in enemies_alive]
        return enemies_alive[np.argmin(distances)]

    def get_state(self):
        """AI核心接口1：获取增强的33维归一化游戏状态，包含更多战术信息"""
        state = np.zeros(33, dtype=np.float32)
        player = self.player
        enemy = self._get_nearest_enemy()
        screen_diag = get_screen_diag()

        # 玩家信息 (5维)
        state[0] = normalize(player.x, 0, SCREEN_WIDTH)
        state[1] = normalize(player.y, 0, SCREEN_HEIGHT)
        state[2] = normalize(player.aim_angle, 0, 2*math.pi)
        state[3] = normalize(player.lives, 0, 5)
        state[4] = normalize(player.cooldown, 0, player.cooldown_max)
        
        # 最近敌人信息 (7维)
        if enemy:
            state[5] = normalize(enemy.x, 0, SCREEN_WIDTH)
            state[6] = normalize(enemy.y, 0, SCREEN_HEIGHT)
            dist = distance_between(player.x, player.y, enemy.x, enemy.y)
            state[7] = normalize(dist, 0, screen_diag)
            # 敌人朝向角度和相对角度
            dx = enemy.x - player.x
            dy = enemy.y - player.y
            enemy_angle = math.atan2(-dy, dx) % (2*math.pi)
            state[8] = normalize(enemy_angle, 0, 2*math.pi)
            # 瞄准角度差异
            angle_diff = abs(player.aim_angle - enemy_angle)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
            state[9] = normalize(angle_diff, 0, math.pi)  # 角度差
            state[10] = normalize(enemy.lives, 0, 5)
            state[11] = 1.0  # 敌人存在标记
        else:
            state[5:12] = 0  # 无敌人时全零
            state[11] = 0
        
        # 敌人子弹威胁信息 (12维) - 最多跟踪6颗最近的敌人子弹
        enemy_bullets = [b for b in self.bullets if not b.is_player_bullet]
        enemy_bullets_sorted = sorted(enemy_bullets, 
                                     key=lambda b: distance_between(player.x, player.y, b.x, b.y))
        
        for i in range(6):  # 最多6颗子弹
            if i < len(enemy_bullets_sorted):
                bullet = enemy_bullets_sorted[i]
                bullet_dist = distance_between(player.x, player.y, bullet.x, bullet.y)
                
                # 子弹位置
                state[12 + i*2] = normalize(bullet.x, 0, SCREEN_WIDTH)
                state[13 + i*2] = normalize(bullet.y, 0, SCREEN_HEIGHT)
                
                # 子弹威胁度评估
                if bullet_dist < 200:  # 危险距离内
                    # 计算子弹是否朝向玩家
                    bullet_to_player_x = player.x - bullet.x
                    bullet_to_player_y = player.y - bullet.y
                    bullet_to_player_angle = math.atan2(-bullet_to_player_y, bullet_to_player_x)
                    bullet_angle_diff = abs(bullet.angle - bullet_to_player_angle)
                    bullet_angle_diff = min(bullet_angle_diff, 2*math.pi - bullet_angle_diff)
                    
                    # 威胁度：距离越近、角度越对准，威胁越大
                    threat_level = (1.0 - bullet_dist/200) * (1.0 - bullet_angle_diff/math.pi)
                    state[24 + i] = threat_level
                else:
                    state[24 + i] = 0
            else:
                state[12 + i*2] = -1.0  # 无子弹位置标记
                state[13 + i*2] = -1.0
                state[24 + i] = 0
        
        # 战术位置信息 (3维)
        if enemy:
            # 重新计算距离（避免作用域问题）
            current_dist = distance_between(player.x, player.y, enemy.x, enemy.y)
            
            # 安全区域评估：距离边界的距离
            dist_to_left = player.x
            dist_to_right = SCREEN_WIDTH - player.x
            dist_to_top = player.y
            dist_to_bottom = SCREEN_HEIGHT - player.y
            min_dist_to_edge = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            state[30] = normalize(min_dist_to_edge, 0, min(SCREEN_WIDTH, SCREEN_HEIGHT)/2)
            
            # 战术优势：玩家是否在更好的位置（可以攻击但不容易被反击）
            if 150 <= current_dist <= 400:  # 理想攻击距离
                state[31] = 1.0
            elif current_dist < 150:  # 太近，危险
                state[31] = 0.3
            else:  # 太远
                state[31] = 0.5
        else:
            state[30] = 0
            state[31] = 0
        
        # 时间压力信息 (1维)
        time_pressure = 1.0 - normalize(self.remaining_time, 0, GAME_TIME_LIMIT)
        state[32] = time_pressure  # 时间越少，压力越大
        
        # 适配点：返回list而非np.array，避免AI代码中张量转换报错
        return state.tolist()

    def do_action(self, action):
        """AI核心接口2：执行AI的动作指令（0-7），游戏内部映射执行"""
        if not self.player.alive or self.game_over: return
        dx, dy, d_angle = 0, 0, 0
        shoot = False

        # 适配点：只保留AI定义的0-7动作，移除未使用的ACTION_MOVE_AIM
        if action == ACTION_UP: dy = -1
        elif action == ACTION_DOWN: dy = 1
        elif action == ACTION_LEFT: dx = -1
        elif action == ACTION_RIGHT: dx = 1
        elif action == ACTION_GUN_LEFT: d_angle = deg2rad(3)
        elif action == ACTION_GUN_RIGHT: d_angle = -deg2rad(3)
        elif action == ACTION_SHOOT: shoot = True

        self.player.move(dx, dy)  # 移除墙壁参数
        self.player.rotate_gun(d_angle)
        if shoot:
            bullet = self.player.shoot()
            if bullet:
                self.bullets.append(bullet)
    
    def do_actions(self, actions):
        """新增：同时执行多个动作（用于手动游玩）"""
        if not self.player.alive or self.game_over: return
        
        dx, dy, d_angle = 0, 0, 0
        shoot = False
        
        # 处理移动组合
        if ACTION_UP in actions:
            dy -= 1
        if ACTION_DOWN in actions:
            dy += 1
        if ACTION_LEFT in actions:
            dx -= 1
        if ACTION_RIGHT in actions:
            dx += 1
            
        # 处理炮管旋转
        if ACTION_GUN_LEFT in actions:
            d_angle += deg2rad(3)
        if ACTION_GUN_RIGHT in actions:
            d_angle -= deg2rad(3)
            
        # 处理射击
        if ACTION_SHOOT in actions:
            shoot = True

        self.player.move(dx, dy)  # 移除墙壁参数
        self.player.rotate_gun(d_angle)
        if shoot:
            bullet = self.player.shoot()
            if bullet:
                self.bullets.append(bullet)

    def _update_enemies(self):
        """【彻底修复】敌方AI炮管精准锁定玩家，炮管朝向和子弹飞行方向完全一致"""
        for enemy in self.enemies:
            if not enemy.alive or self.game_over: continue
            
            # ========== 核心修复：瞄准角度计算 ==========
            dx = self.player.x - enemy.x
            dy = self.player.y - enemy.y
            target_angle = math.atan2(-dy, dx) % (2 * math.pi)
            enemy.aim_angle = target_angle

            # 敌方随机移动
            if random.random() < 0.3:
                enemy.move(random.choice([-1,0,1]), random.choice([-1,0,1]))

            # 敌方射击：限制子弹数量，轻微散布
            current_enemy_bullets = len([b for b in self.bullets if not b.is_player_bullet])
            if current_enemy_bullets < ENEMY_MAX_BULLETS and random.random() < 0.05:
                bullet = enemy.shoot()
                if bullet:
                    bullet.angle += random.uniform(-ENEMY_BULLET_ANGLE_OFFSET, ENEMY_BULLET_ANGLE_OFFSET)
                    self.bullets.append(bullet)
            enemy.update_cooldown()

    def _update_bullets(self):
        """游戏内部：子弹更新和碰撞检测"""
        if self.game_over: return
        for bullet in self.bullets:
            bullet.move()  # 移除墙壁参数
        
        new_bullets = []
        for bullet in self.bullets:
            if not bullet.active: continue
            hit = False
            if bullet.is_player_bullet:
                for enemy in self.enemies:
                    if not enemy.alive: continue
                    if distance_between(bullet.x, bullet.y, enemy.x, enemy.y) < enemy.size//2:
                        enemy.lives -= 1
                        if enemy.lives <= 0:
                            enemy.alive = False
                            self.score += 70  # 适配AI击杀奖励判定（70分=击杀）
                        hit = True
                        break
            else:
                if distance_between(bullet.x, bullet.y, self.player.x, self.player.y) < self.player.size//2:
                    self.player.lives -= 1
                    hit = True
                    if self.player.lives <= 0:
                        self.player.alive = False
                        self.game_over = True
            if not hit:
                new_bullets.append(bullet)
        self.bullets = new_bullets
        
        # 补充敌方坦克
        enemies_alive = [e for e in self.enemies if e.alive]
        if len(enemies_alive) < 3 and random.random() < 0.01 and not self.game_over:
            self.enemies.append(Tank(random.randint(100,700), random.randint(100,200), BLUE))

    def _update_timer(self):
        """新增：更新游戏计时，时间到则清零玩家生命值并结束游戏"""
        if self.game_over or self.remaining_time <= 0:
            return
        self.total_frames += 1
        if self.total_frames % FPS == 0:
            self.remaining_time -= 1
            if self.remaining_time <= 0:
                self.player.lives = 0
                self.player.alive = False
                self.game_over = True

    def step(self):
        """游戏内部：单步更新游戏逻辑（AI调用后，更新游戏状态），返回(reward, done)"""
        if self.game_over: return 0, True
        
        if self.is_render:
            # 渲染模式：正常更新
            self._update_timer()
            self.step_count += 1
            self.player.update_cooldown()
            
            # AI自动开火逻辑
            if self.player.auto_shoot and self.player.alive and self.player.cooldown == 0:
                enemy = self._get_nearest_enemy()
                if enemy and enemy.alive:
                    dx = enemy.x - self.player.x
                    dy = enemy.y - self.player.y
                    target_angle = math.atan2(-dy, dx) % (2*math.pi)
                    angle_diff = abs(self.player.aim_angle - target_angle)
                    angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                    if angle_diff < math.pi/12:
                        bullet = self.player.shoot()
                        if bullet:
                            self.bullets.append(bullet)
            
            self._update_enemies()
            self._update_bullets()
            self.render()  # 调用公开的render方法
        else:
            # 训练模式：快速更新，跳过渲染
            self._update_no_render()
        
        # === 彻底重构奖励函数 - 解决根本矛盾 ===
        reward = 0
        
        # 1. 核心事件奖励（统一score和reward信号）
        score_delta = self.score - self.last_score
        if score_delta > 0:
            reward += score_delta * 5  # 正面事件：击中敌人+35分，击杀+350分
        
        # 2. 生存时间奖励（替代每步惩罚）
        if self.player.alive:
            reward += 0.1  # 生存奖励，鼓励存活更久
        
        # 3. 被击中惩罚（适度，不过分打击探索）
        if self.player.lives < self.last_lives:
            reward -= 5  # 适度惩罚
            self.last_lives = self.player.lives
        
        # 4. 移动奖励（新增 - 解决移动头停滞）
        current_pos = (self.player.x, self.player.y)
        if not hasattr(self, 'last_position'):
            self.last_position = current_pos
        
        movement_distance = distance_between(current_pos[0], current_pos[1], 
                                          self.last_position[0], self.last_position[1])
        if movement_distance > 1:  # 有明显移动
            reward += 0.05  # 鼓励移动探索
        self.last_position = current_pos
        
        # 5. 战术位置奖励（鼓励接近敌人但保持安全距离）
        enemy = self._get_nearest_enemy()
        if enemy and enemy.alive:
            dist = distance_between(self.player.x, self.player.y, enemy.x, enemy.y)
            # 黄金距离区间：150-300像素
            if 150 <= dist <= 300:
                reward += 0.2  # 战术位置奖励
            elif dist < 150:  # 太近了，危险
                reward -= 0.1
            elif dist <= 450:  # 仍在有效攻击范围内
                reward += 0.05
        
        # 6. 瞄准精度奖励（适度奖励，不主导）
        if enemy and enemy.alive:
            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            target_angle = math.atan2(-dy, dx) % (2*math.pi)
            angle_diff = abs(self.player.aim_angle - target_angle)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
            # 精度越高奖励越大
            if angle_diff < math.pi/36:  # 5度内
                reward += 0.1
            elif angle_diff < math.pi/18:  # 10度内
                reward += 0.05
                
        # 7. 射击时机奖励（鼓励有效射击）
        if enemy and enemy.alive:
            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            target_angle = math.atan2(-dy, dx) % (2*math.pi)
            angle_diff = abs(self.player.aim_angle - target_angle)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
            # 只有瞄准较好时射击才给奖励
            if angle_diff < math.pi/12 and len([b for b in self.bullets if b.is_player_bullet]) < 3:
                if self.player.cooldown > 0 and self.player.cooldown == self.player.cooldown_max - 1:
                    # 刚刚射击，且瞄准较好
                    reward += 0.3
        
        self.last_score = self.score
        # 游戏结束条件
        done = self.game_over or self.step_count >= self.max_steps or self.score >= 1000
        return reward, done

    # 修复点2：定义公开的render方法，供AI代码调用，内部调用私有渲染逻辑
    def render(self):
        """AI公开调用接口：画面渲染，解决bool不可调用核心错误"""
        self._render()

    def _draw_game_over(self):
        """绘制游戏结束界面+重新开始按钮（新增时间耗尽提示）"""
        if not self.is_render: return
        mask = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 200))
        self.screen.blit(mask, (0, 0))
        # 区分游戏结束原因：时间耗尽/生命值归零
        if self.remaining_time <= 0:
            game_over_text = self.big_font.render("TIME UP!", True, RED_ALERT)
        else:
            game_over_text = self.big_font.render("GAME OVER", True, RED)
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
        self.screen.blit(game_over_text, text_rect)
        # 重新开始按钮
        mouse_pos = pygame.mouse.get_pos()
        btn_color = BUTTON_HOVER if self.restart_btn.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, btn_color, self.restart_btn, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, self.restart_btn, 3, border_radius=10)
        btn_text = self.button_font.render("重新开始", True, WHITE)
        btn_text_rect = btn_text.get_rect(center=self.restart_btn.center)
        self.screen.blit(btn_text, btn_text_rect)
        # 最终分数
        score_text = self.small_font.render(f"最终分数: {self.score}", True, YELLOW)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 130))
        self.screen.blit(score_text, score_rect)

    def _render(self):
        """游戏内部私有渲染逻辑，所有渲染判断用重命名后的is_render"""
        if not self.is_render or not self.screen: return
        self.screen.fill(BLACK)
        if not self.game_over:
            # 不再绘制墙体
            # 绘制坦克和子弹
            self.player.draw(self.screen)
            for enemy in self.enemies:
                enemy.draw(self.screen)
            for bullet in self.bullets:
                bullet.draw(self.screen)
            # 绘制信息栏
            self.screen.blit(self.small_font.render(f"Score: {self.score}", True, WHITE), (10,10))
            self.screen.blit(self.small_font.render(f"Lives: {self.player.lives}", True, RED), (SCREEN_WIDTH-120,10))
            enemy_bullet_num = len([b for b in self.bullets if not b.is_player_bullet])
            self.screen.blit(self.small_font.render(f"Enemy Bullets: {enemy_bullet_num}/{ENEMY_MAX_BULLETS}", True, YELLOW), (200,10))
            # 绘制剩余时间（警示色）
            time_color = RED_ALERT if self.remaining_time <= 10 else WHITE
            self.screen.blit(self.small_font.render(f"Time: {self.remaining_time}s", True, time_color), (450,10))
        else:
            self._draw_game_over()
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def _update_no_render(self):
        """训练时的快速更新，跳过渲染和帧率限制"""
        if self.game_over: return
        self.step_count += 1
        self.player.update_cooldown()
        
        # AI自动开火逻辑
        if self.player.auto_shoot and self.player.alive and self.player.cooldown == 0:
            enemy = self._get_nearest_enemy()
            if enemy and enemy.alive:
                dx = enemy.x - self.player.x
                dy = enemy.y - self.player.y
                target_angle = math.atan2(-dy, dx) % (2*math.pi)
                angle_diff = abs(self.player.aim_angle - target_angle)
                angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                if angle_diff < math.pi/12:
                    bullet = self.player.shoot()
                    if bullet:
                        self.bullets.append(bullet)
        
        self._update_enemies()
        self._update_bullets()
        self._update_timer()

    def _check_restart_click(self):
        """检测重新开始按钮点击"""
        if self.game_over and pygame.mouse.get_pressed()[0]:
            if self.restart_btn.collidepoint(pygame.mouse.get_pos()):
                self.reset()

    def enable_auto_shoot(self):
        """启用AI自动开火功能（瞄准后自动开火）"""
        self.player.set_auto_shoot(True)
    
    def disable_auto_shoot(self):
        """禁用AI自动开火功能"""
        self.player.set_auto_shoot(False)

    def manual_play(self):
        """独立运行：人类手动玩游戏（含重新开始交互）"""
        print("开始手动玩游戏！W/A/S/D移动，←/→转炮管，空格射击，Q退出")
        print("提示：现在可以同时按下多个按键了！")
        print("例如：W+D向右上移动，同时按←/→旋转炮管，同时按空格射击")
        self.reset()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    return
            self._check_restart_click()
            if not self.game_over:
                keys = pygame.key.get_pressed()
                actions = []
                # 移动按键检测 - 可以同时按
                if keys[pygame.K_w]: actions.append(ACTION_UP)
                if keys[pygame.K_s]: actions.append(ACTION_DOWN)
                if keys[pygame.K_a]: actions.append(ACTION_LEFT)
                if keys[pygame.K_d]: actions.append(ACTION_RIGHT)
                # 瞄准按键检测 - 可以和移动同时按
                if keys[pygame.K_LEFT]: actions.append(ACTION_GUN_LEFT)
                if keys[pygame.K_RIGHT]: actions.append(ACTION_GUN_RIGHT)
                # 射击按键检测 - 可以和移动、瞄准同时按
                if keys[pygame.K_SPACE]: actions.append(ACTION_SHOOT)
                
                # 执行多个动作
                if actions:
                    self.do_actions(actions)
                else:
                    # 如果没有按任何键，坦克保持静止
                    pass
            self.step()

# ====================== 独立运行游戏（测试用） =======================
if __name__ == "__main__":
    game = TankGame(render=True)
    game.manual_play()