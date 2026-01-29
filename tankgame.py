import pygame
import random
import math
import numpy as np

# ====================== 游戏常量（可独立修改，不影响AI） =======================
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
TANK_SIZE = 40
BULLET_RADIUS = 5
WALL_SIZE = 27
TANK_SPEED = 4
BULLET_BASE_SPEED = 14
MAX_BOUNCES = 10
ENEMY_BULLET_ANGLE_OFFSET = math.pi/60  # 射击轻微散布，不影响瞄准
ENEMY_MAX_BULLETS = 9
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

# 动作常量（和AI统一，0-8，游戏只负责映射执行）
ACTION_IDLE = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_GUN_LEFT = 5
ACTION_GUN_RIGHT = 6
ACTION_SHOOT = 7
ACTION_MOVE_AIM = 8

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

    def move(self, walls):
        if not self.active: return
        next_x = self.x + math.cos(self.angle) * self.speed
        next_y = self.y - math.sin(self.angle) * self.speed
        bullet_rect = pygame.Rect(
            next_x - self.radius, next_y - self.radius,
            self.radius*2, self.radius*2
        )
        wall_collide = False
        collide_x = False
        collide_y = False
        for wall in walls:
            wall_rect = pygame.Rect(wall[0], wall[1], WALL_SIZE, WALL_SIZE)
            if bullet_rect.colliderect(wall_rect):
                wall_collide = True
                collide_x = abs(bullet_rect.centerx - wall_rect.centerx) < (WALL_SIZE/2 + self.radius)
                collide_y = abs(bullet_rect.centery - wall_rect.centery) < (WALL_SIZE/2 + self.radius)
                break
        if wall_collide:
            if collide_x:
                self.angle = -self.angle
            if collide_y:
                self.angle = math.pi - self.angle
            self.bounce_count += 1
            if self.bounce_count >= self.max_bounces:
                self.active = False
            return
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

    def draw(self, screen):
        if not self.alive: return
        rect = pygame.Rect(self.x-self.size//2, self.y-self.size//2, self.size, self.size)
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        # 炮管绘制和子弹发射用完全相同的角度计算，保证朝向和子弹一致
        gun_x = self.x + math.cos(self.aim_angle) * 25
        gun_y = self.y - math.sin(self.aim_angle) * 25
        pygame.draw.line(screen, BLACK, (self.x, self.y), (gun_x, gun_y), 5)

    def move(self, dx, dy, walls):
        if not self.alive: return
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        new_x = max(self.size//2, min(SCREEN_WIDTH - self.size//2, new_x))
        new_y = max(self.size//2, min(SCREEN_HEIGHT - self.size//2, new_y))
        tank_rect = pygame.Rect(new_x-self.size//2, new_y-self.size//2, self.size, self.size)
        for wall in walls:
            wall_rect = pygame.Rect(wall[0], wall[1], WALL_SIZE, WALL_SIZE)
            if tank_rect.colliderect(wall_rect):
                return
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

# ====================== 核心游戏类（暴露AI接口，可独立运行） =======================
class TankGame:
    def __init__(self, render=True):
        self.render = render
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) if self.render else None
        self.clock = pygame.time.Clock()
        self.small_font = pygame.font.SysFont(None, 36) if self.render else None
        self.big_font = pygame.font.SysFont(None, 80) if self.render else None
        self.button_font = pygame.font.SysFont(None, 48) if self.render else None
        if self.render:
            pygame.display.set_caption("坦克大战 - 带时间限制+敌方精准瞄准")
        self.restart_btn = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 50, 200, 60)
        self.game_over = False
        self.reset()

    def reset(self):
        """重置游戏，AI调用：初始化所有游戏对象（含时间计时）"""
        self.player = Tank(SCREEN_WIDTH//2, SCREEN_HEIGHT-TANK_SIZE-20, GREEN, is_player=True)
        self.enemies = [Tank(random.randint(100,700), random.randint(100,200), BLUE) for _ in range(3)]
        self.walls = self._generate_walls()
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

    def _generate_walls(self):
        walls = []
        player_rect = pygame.Rect(self.player.x-60, self.player.y-60, 120, 120)
        for _ in range(20):
            x = random.randint(0, (SCREEN_WIDTH-WALL_SIZE)//WALL_SIZE)*WALL_SIZE
            y = random.randint(0, (SCREEN_HEIGHT-WALL_SIZE*3)//WALL_SIZE)*WALL_SIZE
            wall_rect = pygame.Rect(x, y, WALL_SIZE, WALL_SIZE)
            if not wall_rect.colliderect(player_rect):
                walls.append((x, y))
        return walls

    def _get_nearest_enemy(self):
        if not self.enemies: return None
        distances = [distance_between(self.player.x, self.player.y, e.x, e.y) for e in self.enemies]
        return self.enemies[np.argmin(distances)]

    def get_state(self):
        """AI核心接口1：获取14维归一化游戏状态，返回np.array"""
        state = np.zeros(14, dtype=np.float32)
        player = self.player
        enemy = self._get_nearest_enemy()
        screen_diag = get_screen_diag()

        state[0] = normalize(player.x, 0, SCREEN_WIDTH)
        state[1] = normalize(player.y, 0, SCREEN_HEIGHT)
        state[2] = normalize(player.aim_angle, 0, 2*math.pi)
        if enemy:
            state[3] = normalize(enemy.x, 0, SCREEN_WIDTH)
            state[4] = normalize(enemy.y, 0, SCREEN_HEIGHT)
            dist = distance_between(player.x, player.y, enemy.x, enemy.y)
            state[5] = normalize(dist, 0, screen_diag)
            # 玩家朝向敌人的角度也用相同计算逻辑，保证AI状态数据准确
            dx = enemy.x - player.x
            dy = enemy.y - player.y
            enemy_angle = math.atan2(-dy, dx) % (2*math.pi)
            state[6] = normalize(enemy_angle, 0, 2*math.pi)
            state[11] = normalize(enemy.lives, 0, 5)
        state[7] = normalize(player.cooldown, 0, player.cooldown_max)
        player_bullets = [b for b in self.bullets if b.is_player_bullet]
        state[8] = normalize(len(player_bullets), 0, 10)
        enemy_bullets = [b for b in self.bullets if not b.is_player_bullet]
        state[9] = normalize(len(enemy_bullets), 0, 10)
        state[10] = normalize(player.lives, 0, 5)
        wall_dists = [distance_between(player.x, player.y, w[0]+WALL_SIZE//2, w[1]+WALL_SIZE//2) for w in self.walls]
        state[12] = normalize(min(wall_dists), 0, screen_diag) if wall_dists else 1.0
        state[13] = normalize(self.score, 0, 1000)
        return state

    def do_action(self, action):
        """AI核心接口2：执行AI的动作指令（0-8），游戏内部映射执行"""
        if not self.player.alive or self.game_over: return
        dx, dy, d_angle = 0, 0, 0
        shoot = False

        if action == ACTION_UP: dy = -1
        elif action == ACTION_DOWN: dy = 1
        elif action == ACTION_LEFT: dx = -1
        elif action == ACTION_RIGHT: dx = 1
        elif action == ACTION_GUN_LEFT: d_angle = deg2rad(3)
        elif action == ACTION_GUN_RIGHT: d_angle = -deg2rad(3)
        elif action == ACTION_SHOOT: shoot = True
        elif action == ACTION_MOVE_AIM:
            enemy = self._get_nearest_enemy()
            if enemy:
                dx, dy = enemy.x - self.player.x, enemy.y - self.player.y
                dx, dy = normalize_vector(dx, dy)
                dx, dy = int(np.sign(dx)), int(np.sign(dy))
                # 玩家自动瞄准也用相同逻辑
                enemy_angle = math.atan2(-dy, dx) % (2*math.pi)
                d_angle = enemy_angle - self.player.aim_angle
                d_angle = np.clip(d_angle, -deg2rad(3), deg2rad(3))

        self.player.move(dx, dy, self.walls)
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
            # 计算敌方到玩家的向量（dx: x轴差值, dy: y轴差值）
            dx = self.player.x - enemy.x  # 敌方x → 玩家x
            dy = self.player.y - enemy.y  # 敌方y → 玩家y
            # 适配pygame坐标系的瞄准角度：math.atan2(-dy, dx)
            # 该计算和子弹发射/炮管绘制的cos/sin计算完全匹配，保证朝向绝对正确
            target_angle = math.atan2(-dy, dx) % (2 * math.pi)
            enemy.aim_angle = target_angle  # 炮管直接锁定玩家，无偏移

            # 敌方随机移动（保留原有逻辑）
            if random.random() < 0.3:
                enemy.move(random.choice([-8,0,8]), random.choice([-8,0,8]), self.walls)

            # 敌方射击：限制3颗子弹，射击时轻微散布（保留原有逻辑）
            current_enemy_bullets = len([b for b in self.bullets if not b.is_player_bullet])
            if current_enemy_bullets < ENEMY_MAX_BULLETS and random.random() < 1:
                bullet = enemy.shoot()
                if bullet:
                    # 仅在射击时添加轻微角度散布，不影响炮管瞄准
                    bullet.angle += random.uniform(-ENEMY_BULLET_ANGLE_OFFSET, ENEMY_BULLET_ANGLE_OFFSET)
                    self.bullets.append(bullet)
            enemy.update_cooldown()

    def _update_bullets(self):
        """游戏内部：子弹更新和碰撞检测"""
        if self.game_over: return
        for bullet in self.bullets:
            bullet.move(self.walls)
        
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
                            self.score += 20
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
        
        self.enemies = [e for e in self.enemies if e.alive]
        if len(self.enemies) < 3 and random.random() < 0.01 and not self.game_over:
            self.enemies.append(Tank(random.randint(100,700), random.randint(100,200), BLUE))

    def _update_timer(self):
        """新增：更新游戏计时，时间到则清零玩家生命值并结束游戏"""
        if self.game_over or self.remaining_time <= 0:
            return
        # 累计总帧数，每60帧=1秒（FPS固定60）
        self.total_frames += 1
        if self.total_frames % FPS == 0:
            self.remaining_time -= 1
            # 时间到：清零玩家生命值，标记游戏结束
            if self.remaining_time <= 0:
                self.player.lives = 0
                self.player.alive = False
                self.game_over = True

    def step(self):
        """游戏内部：单步更新游戏逻辑（AI调用后，更新游戏状态）"""
        if self.game_over: return 0, True
        # 新增：先更新计时
        self._update_timer()
        self.step_count += 1
        self.player.update_cooldown()
        self._update_enemies()
        self._update_bullets()
        self._render()
        reward = 0
        reward += self.score - self.last_score
        if self.player.lives < self.last_lives:
            reward -= 10
            self.last_lives = self.player.lives
        self.last_score = self.score
        # 游戏结束条件：玩家死亡/步数超限/分数达标/时间耗尽
        done = self.game_over or self.step_count >= self.max_steps or self.score >= 500
        return reward, done

    def _draw_game_over(self):
        """绘制游戏结束界面+重新开始按钮（新增时间耗尽提示）"""
        if not self.render: return
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
        """游戏内部：画面渲染（含游戏结束界面+新增时间显示）"""
        if not self.render or not self.screen: return
        self.screen.fill(BLACK)
        if not self.game_over:
            for wall in self.walls:
                pygame.draw.rect(self.screen, GRAY, (wall[0], wall[1], WALL_SIZE, WALL_SIZE))
                pygame.draw.rect(self.screen, BLACK, (wall[0], wall[1], WALL_SIZE, WALL_SIZE), 1)
            self.player.draw(self.screen)
            for enemy in self.enemies:
                enemy.draw(self.screen)
            for bullet in self.bullets:
                bullet.draw(self.screen)
            # 绘制信息栏（新增时间显示）
            self.screen.blit(self.small_font.render(f"Score: {self.score}", True, WHITE), (10,10))
            self.screen.blit(self.small_font.render(f"Lives: {self.player.lives}", True, RED), (SCREEN_WIDTH-120,10))
            enemy_bullet_num = len([b for b in self.bullets if not b.is_player_bullet])
            self.screen.blit(self.small_font.render(f"Enemy Bullets: {enemy_bullet_num}/{ENEMY_MAX_BULLETS}", True, YELLOW), (200,10))
            # 新增：绘制剩余时间，剩余10秒内显示警示红色
            time_color = RED_ALERT if self.remaining_time <= 10 else WHITE
            self.screen.blit(self.small_font.render(f"Time: {self.remaining_time}s", True, time_color), (450,10))
        else:
            self._draw_game_over()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _check_restart_click(self):
        """检测重新开始按钮点击"""
        if self.game_over and pygame.mouse.get_pressed()[0]:
            if self.restart_btn.collidepoint(pygame.mouse.get_pos()):
                self.reset()

    def manual_play(self):
        """独立运行：人类手动玩游戏（含重新开始交互）"""
        print("开始手动玩游戏！W/A/S/D移动，←/→转炮管，空格射击，Q退出")
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
                action = ACTION_IDLE
                if keys[pygame.K_w]: action = ACTION_UP
                elif keys[pygame.K_s]: action = ACTION_DOWN
                elif keys[pygame.K_a]: action = ACTION_LEFT
                elif keys[pygame.K_d]: action = ACTION_RIGHT
                elif keys[pygame.K_LEFT]: action = ACTION_GUN_LEFT
                elif keys[pygame.K_RIGHT]: action = ACTION_GUN_RIGHT
                elif keys[pygame.K_SPACE]: action = ACTION_SHOOT
                self.do_action(action)
            self.step()

# ====================== 独立运行游戏（测试用） =======================
if __name__ == "__main__":
    game = TankGame(render=True)
    game.manual_play()