import pygame
import random
import sys
import math

# ====================== 先定义工具函数（核心修正）======================
def deg2rad(deg):
    """角度转弧度"""
    return deg * math.pi / 180.0

def rad2deg(rad):
    """弧度转角度"""
    return rad * 180.0 / math.pi

def normalize_vector(dx, dy):
    """向量归一化，避免速度随距离变化"""
    length = math.hypot(dx, dy)
    if length == 0:
        return 0, 0
    return dx / length, dy / length

# ====================== 再定义游戏常量（此时函数已存在）======================
# 初始化Pygame
pygame.init()

# 游戏常量设置
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
TANK_SIZE = 40
BULLET_RADIUS = 6  # 子弹改为圆形，用半径表示
WALL_SIZE = 40
TANK_SPEED = 4
BULLET_SPEED = 10
MAX_BOUNCES = 4   # 子弹最大反弹次数（含边框反弹）
ENEMY_BULLET_ANGLE_OFFSET = deg2rad(5)  # 敌方三发子弹角度偏移量（左右各5°）

# 按钮常量
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 60
BUTTON_COLOR = (0, 150, 255)
BUTTON_HOVER_COLOR = (0, 200, 255)
BUTTON_TEXT_COLOR = (255, 255, 255)

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# 设置屏幕
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("坦克大战（360°转向+物理反弹+敌方锁定瞄准）")
clock = pygame.time.Clock()
pygame.mouse.set_visible(True)  # 显示鼠标，用于瞄准

# ====================== 坦克类 ======================
class Tank:
    def __init__(self, x, y, color, is_player=False):
        self.x = x  # 坦克中心x坐标（方便计算角度）
        self.y = y  # 坦克中心y坐标
        self.color = color
        self.is_player = is_player
        self.size = TANK_SIZE
        self.speed = TANK_SPEED if is_player else 2
        self.lives = 3 if is_player else 1
        self.cooldown = 0
        self.cooldown_max = 25 if is_player else 40  # 敌方冷却稍长，避免太密集
        # 360°转向相关
        self.aim_angle = 0.0  # 炮管瞄准角度（弧度，0为正右，逆时针增加）
        self.move_dir = (0, 0)  # 移动方向向量(x, y)，由WASD控制

    def draw(self):
        # 坦克主体：以中心为原点画正方形，适配360°绘制
        tank_rect = pygame.Rect(
            self.x - self.size//2,
            self.y - self.size//2,
            self.size,
            self.size
        )
        pygame.draw.rect(screen, self.color, tank_rect)
        pygame.draw.rect(screen, BLACK, tank_rect, 2)  # 描边
        
        # 炮管：根据瞄准角度绘制，长度25，360°随鼠标/玩家位置旋转
        gun_length = 25
        gun_x = self.x + math.cos(self.aim_angle) * gun_length
        gun_y = self.y - math.sin(self.aim_angle) * gun_length  # pygamey轴向下，故减sin
        pygame.draw.line(screen, BLACK, (self.x, self.y), (gun_x, gun_y), 5)

    def update_aim(self, target_x, target_y):
        """根据目标坐标更新炮管瞄准角度（360°实时更新）"""
        dx = target_x - self.x
        dy = target_y - self.y
        # 计算与正右方向的夹角（弧度），pygamey轴向下，需调整符号
        self.aim_angle = math.atan2(-dy, dx)
        self.aim_angle %= (2 * math.pi)  # 角度归一化到0-2π

    def update_move_dir(self, keys):
        """修改为WASD控制移动方向向量（替代方向键）"""
        dx, dy = 0, 0
        if keys[pygame.K_w]:  # W键向上
            dy -= 1
        if keys[pygame.K_s]:  # S键向下
            dy += 1
        if keys[pygame.K_a]:  # A键向左
            dx -= 1
        if keys[pygame.K_d]:  # D键向右
            dx += 1
        # 向量归一化，避免斜向移动速度过快
        self.move_dir = normalize_vector(dx, dy)

    def move(self, walls):
        """根据移动方向向量移动，带墙体和边界碰撞检测"""
        if self.move_dir == (0, 0):
            return  # 无移动方向则不移动
        
        # 计算新位置
        new_x = self.x + self.move_dir[0] * self.speed
        new_y = self.y + self.move_dir[1] * self.speed
        
        # 边界检测：保证坦克整体在屏幕内
        half_size = self.size // 2
        new_x = max(half_size, min(SCREEN_WIDTH - half_size, new_x))
        new_y = max(half_size, min(SCREEN_HEIGHT - half_size, new_y))
        
        # 墙体碰撞检测：坦克矩形与墙体矩形碰撞则不移动
        tank_rect = pygame.Rect(
            new_x - half_size,
            new_y - half_size,
            self.size,
            self.size
        )
        for wall in walls:
            wall_rect = pygame.Rect(wall[0], wall[1], WALL_SIZE, WALL_SIZE)
            if tank_rect.colliderect(wall_rect):
                return  # 碰撞则取消移动
        
        # 无碰撞则更新位置
        self.x = new_x
        self.y = new_y

    def shoot(self, is_three_shot=False):
        """
        发射子弹，支持单发/三发散射
        is_three_shot: 敌方专用，是否发射三发散射子弹
        返回：子弹列表
        """
        if self.cooldown > 0:
            return []
        self.cooldown = self.cooldown_max
        bullets = []
        # 基础发射角度（玩家/敌方单发）
        base_angles = [self.aim_angle]
        # 敌方三发散射：中心角度±偏移量
        if is_three_shot:
            base_angles = [
                self.aim_angle - ENEMY_BULLET_ANGLE_OFFSET,
                self.aim_angle,
                self.aim_angle + ENEMY_BULLET_ANGLE_OFFSET
            ]
        
        # 生成对应角度的子弹
        for angle in base_angles:
            angle %= (2 * math.pi)  # 角度归一化
            # 子弹初始位置：炮管末端，避免贴坦克发射
            bullet_x = self.x + math.cos(angle) * (self.size//2 + BULLET_RADIUS)
            bullet_y = self.y - math.sin(angle) * (self.size//2 + BULLET_RADIUS)
            bullets.append(Bullet(bullet_x, bullet_y, angle, self.is_player))
        return bullets

    def update_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1

# ====================== 子弹类（新增边框反弹）=====================
class Bullet:
    def __init__(self, x, y, angle, is_player_bullet):
        self.x = x  # 子弹中心x坐标
        self.y = y  # 子弹中心y坐标
        self.angle = angle  # 子弹飞行角度（弧度），核心：用角度替代离散方向
        self.speed = BULLET_SPEED
        self.is_player_bullet = is_player_bullet
        self.radius = BULLET_RADIUS
        self.bounce_count = 0
        self.max_bounces = MAX_BOUNCES
        # 子弹碰撞矩形（方便与墙体检测）
        self.rect = self._get_rect()

    def _get_rect(self):
        """根据中心坐标和半径获取碰撞矩形"""
        return pygame.Rect(
            self.x - self.radius,
            self.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )

    def draw(self):
        """绘制圆形子弹"""
        color = YELLOW if self.is_player_bullet else RED
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius, 1)

    def move(self):
        """根据飞行角度移动，实时更新碰撞矩形"""
        self.x += math.cos(self.angle) * self.speed
        self.y -= math.sin(self.angle) * self.speed  # pygamey轴向下，减sin
        self.rect = self._get_rect()

    def bounce(self, wall_rect):
        """
        物理反弹核心：入射角=反射角
        wall_rect：墙体/边框的碰撞矩形
        返回：True=反弹成功，False=超过最大反弹次数
        """
        if self.bounce_count >= self.max_bounces:
            return False
        
        # 确定碰撞的边（上/下/左/右），核心：判断子弹与矩形的重叠区域
        overlap_top = self.rect.bottom - wall_rect.top
        overlap_bottom = wall_rect.bottom - self.rect.top
        overlap_left = self.rect.right - wall_rect.left
        overlap_right = wall_rect.right - self.rect.left

        # 找到最小重叠的边，即为碰撞边（优先级：上下/左右，避免斜角碰撞误判）
        min_overlap = min(overlap_top, overlap_bottom, overlap_left, overlap_right)
        
        if min_overlap in (overlap_top, overlap_bottom):
            # 碰撞上下边：y轴方向反射，角度取π-原角度（入射角=反射角）
            self.angle = math.pi - self.angle
            # 修正位置：避免子弹卡在墙/边框内
            if min_overlap == overlap_top:
                self.y = wall_rect.top - self.radius - 1
            else:
                self.y = wall_rect.bottom + self.radius + 1
        else:
            # 碰撞左右边：x轴方向反射，角度取-原角度（入射角=反射角）
            self.angle = -self.angle
            # 修正位置：避免子弹卡在墙/边框内
            if min_overlap == overlap_left:
                self.x = wall_rect.left - self.radius - 1
            else:
                self.x = wall_rect.right + self.radius + 1

        self.angle %= (2 * math.pi)  # 反弹后角度归一化
        self.bounce_count += 1
        self.rect = self._get_rect()  # 更新修正后的碰撞矩形
        return True

    def check_border_bounce(self):
        """检测子弹是否碰撞屏幕边框，若碰撞则反弹"""
        # 定义屏幕边框的碰撞矩形
        border_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        # 检查子弹是否超出边框（碰撞）
        if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH or self.rect.top < 0 or self.rect.bottom > SCREEN_HEIGHT:
            return self.bounce(border_rect)
        return True  # 未碰撞边框，返回True表示子弹仍有效

    def is_out_of_bounds(self):
        """判断子弹是否超出屏幕边界过远（反弹次数用尽后移除）"""
        return (self.x < -self.radius * 2 or self.x > SCREEN_WIDTH + self.radius * 2 or
                self.y < -self.radius * 2 or self.y > SCREEN_HEIGHT + self.radius * 2)

# ====================== 游戏主类 ======================
class TankGame:
    def __init__(self):
        self.reset_game()  # 初始化/重置游戏状态

    def reset_game(self):
        """重置所有游戏状态（用于重新开始）"""
        # 玩家坦克初始位置：屏幕下方中间（中心坐标）
        self.player_tank = Tank(
            SCREEN_WIDTH//2,
            SCREEN_HEIGHT - TANK_SIZE - 20,
            GREEN,
            is_player=True
        )
        self.enemy_tanks = []
        self.bullets = []
        self.walls = self._generate_walls()
        self.score = 0
        self.game_over = False
        self.spawn_enemy_timer = 0
        self.spawn_enemy_interval = 100  # 敌方生成间隔
        self.max_enemies = 8  # 最大同时存在敌方坦克数

    def _generate_walls(self):
        """生成随机墙体，避免与玩家初始位置重叠"""
        walls = []
        player_half = self.player_tank.size // 2
        player_rect = pygame.Rect(
            self.player_tank.x - player_half * 3,
            self.player_tank.y - player_half * 3,
            player_half * 6,
            player_half * 6
        )
        for _ in range(18):
            x = random.randint(0, (SCREEN_WIDTH - WALL_SIZE) // WALL_SIZE) * WALL_SIZE
            y = random.randint(0, (SCREEN_HEIGHT - WALL_SIZE * 3) // WALL_SIZE) * WALL_SIZE
            wall_rect = pygame.Rect(x, y, WALL_SIZE, WALL_SIZE)
            if not wall_rect.colliderect(player_rect):
                walls.append((x, y))
        return walls

    def spawn_enemy(self):
        """生成敌方坦克，随机初始移动方向"""
        x = random.randint(TANK_SIZE, SCREEN_WIDTH - TANK_SIZE)
        y = random.randint(TANK_SIZE, SCREEN_HEIGHT // 3)
        enemy = Tank(x, y, BLUE)
        # 敌方随机初始移动方向
        enemy.move_dir = normalize_vector(random.uniform(-1,1), random.uniform(-1,1))
        self.enemy_tanks.append(enemy)

    def handle_collisions(self):
        """处理所有碰撞：子弹-墙体/边框（反弹）、子弹-坦克、坦克-坦克"""
        # 1. 子弹-墙体/边框碰撞：物理反弹
        bullets_to_remove = []
        for i, bullet in enumerate(self.bullets):
            collided = False
            # 先检测墙体碰撞
            for wall in self.walls:
                wall_rect = pygame.Rect(wall[0], wall[1], WALL_SIZE, WALL_SIZE)
                if bullet.rect.colliderect(wall_rect):
                    if not bullet.bounce(wall_rect):
                        bullets_to_remove.append(i)
                    collided = True
                    break
            # 未碰撞墙体则检测边框反弹
            if not collided:
                if not bullet.check_border_bounce():
                    bullets_to_remove.append(i)
            # 子弹出界过远则移除
            if bullet.is_out_of_bounds() and i not in bullets_to_remove:
                bullets_to_remove.append(i)
        # 移除反弹失效/出界的子弹
        for idx in sorted(bullets_to_remove, reverse=True):
            del self.bullets[idx]

        # 2. 子弹-坦克碰撞：击中则销毁子弹和坦克
        tanks_to_remove = []
        bullets_to_remove = []
        for i, bullet in enumerate(self.bullets):
            # 玩家子弹击中敌方
            if bullet.is_player_bullet:
                for j, enemy in enumerate(self.enemy_tanks):
                    enemy_rect = pygame.Rect(
                        enemy.x - enemy.size//2,
                        enemy.y - enemy.size//2,
                        enemy.size,
                        enemy.size
                    )
                    if bullet.rect.colliderect(enemy_rect):
                        bullets_to_remove.append(i)
                        tanks_to_remove.append(j)
                        self.score += 20  # 敌方变强，得分提高
                        break
            # 敌方子弹击中玩家
            else:
                player_rect = pygame.Rect(
                    self.player_tank.x - self.player_tank.size//2,
                    self.player_tank.y - self.player_tank.size//2,
                    self.player_tank.size,
                    self.player_tank.size
                )
                if bullet.rect.colliderect(player_rect):
                    bullets_to_remove.append(i)
                    self.player_tank.lives -= 1
                    if self.player_tank.lives <= 0:
                        self.game_over = True
                    break
        # 移除被击中的坦克和子弹
        for idx in sorted(tanks_to_remove, reverse=True):
            del self.enemy_tanks[idx]
        for idx in sorted(bullets_to_remove, reverse=True):
            if idx < len(self.bullets):
                del self.bullets[idx]

        # 3. 敌方坦克-玩家坦克碰撞：玩家扣血
        player_rect = pygame.Rect(
            self.player_tank.x - self.player_tank.size//2,
            self.player_tank.y - self.player_tank.size//2,
            self.player_tank.size,
            self.player_tank.size
        )
        for enemy in self.enemy_tanks:
            enemy_rect = pygame.Rect(
                enemy.x - enemy.size//2,
                enemy.y - enemy.size//2,
                enemy.size,
                enemy.size
            )
            if player_rect.colliderect(enemy_rect):
                self.player_tank.lives -= 1
                self.enemy_tanks.remove(enemy)
                if self.player_tank.lives <= 0:
                    self.game_over = True
                break

    def enemy_ai(self):
        """敌方AI：锁定玩家瞄准+随机移动+三发散射射击"""
        for enemy in self.enemy_tanks:
            # 1. 核心：实时锁定玩家，更新瞄准角度
            enemy.update_aim(self.player_tank.x, self.player_tank.y)
            # 2. 随机调整移动方向，保持机动性
            if random.random() < 0.12:
                enemy.move_dir = normalize_vector(random.uniform(-1,1), random.uniform(-1,1))
            # 3. 移动敌方坦克，带碰撞检测
            enemy.move(self.walls)
            # 4. 随机射击：发射三发散射子弹，敌方专用
            if random.random() < 0.02:
                enemy_bullets = enemy.shoot(is_three_shot=True)
                self.bullets.extend(enemy_bullets)
            # 5. 更新射击冷却
            enemy.update_cooldown()

    def draw_restart_button(self, mouse_pos):
        """绘制重新开始按钮，带hover效果"""
        # 计算按钮居中位置
        button_x = SCREEN_WIDTH // 2 - BUTTON_WIDTH // 2
        button_y = SCREEN_HEIGHT // 2 + 30
        button_rect = pygame.Rect(button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
        
        # 判断鼠标是否在按钮上，切换颜色
        if button_rect.collidepoint(mouse_pos):
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_COLOR
        
        # 绘制按钮
        pygame.draw.rect(screen, color, button_rect, border_radius=10)  # 圆角按钮
        pygame.draw.rect(screen, WHITE, button_rect, 2, border_radius=10)  # 描边
        
        # 绘制按钮文字
        font = pygame.font.SysFont(None, 40)
        text = font.render("重新开始", True, BUTTON_TEXT_COLOR)
        text_x = button_x + (BUTTON_WIDTH - text.get_width()) // 2
        text_y = button_y + (BUTTON_HEIGHT - text.get_height()) // 2
        screen.blit(text, (text_x, text_y))
        
        return button_rect

    def run(self):
        """游戏主循环（支持重新开始）"""
        while True:  # 外层循环，支持多次游戏
            while not self.game_over:  # 单次游戏循环
                clock.tick(FPS)
                mouse_x, mouse_y = pygame.mouse.get_pos()  # 获取实时鼠标坐标

                # 事件处理
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    # 玩家射击：空格键（单发）
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        player_bullets = self.player_tank.shoot()
                        self.bullets.extend(player_bullets)

                # 玩家坦克更新：360°瞄准（鼠标）+ 移动方向（WASD）
                keys = pygame.key.get_pressed()
                self.player_tank.update_aim(mouse_x, mouse_y)  # 实时跟随鼠标转向
                self.player_tank.update_move_dir(keys)         # 更新WASD移动方向
                self.player_tank.move(self.walls)              # 执行移动
                self.player_tank.update_cooldown()             # 更新射击冷却

                # 生成敌方坦克
                self.spawn_enemy_timer += 1
                if self.spawn_enemy_timer >= self.spawn_enemy_interval and len(self.enemy_tanks) < self.max_enemies:
                    self.spawn_enemy()
                    self.spawn_enemy_timer = 0

                # 敌方AI逻辑（锁定玩家+三发射击）
                self.enemy_ai()

                # 所有子弹移动
                for bullet in self.bullets:
                    bullet.move()

                # 碰撞检测（反弹+击中）
                self.handle_collisions()

                # 绘制所有游戏元素
                screen.fill(BLACK)
                # 绘制墙体
                for wall in self.walls:
                    pygame.draw.rect(screen, GRAY, (wall[0], wall[1], WALL_SIZE, WALL_SIZE))
                    pygame.draw.rect(screen, WHITE, (wall[0], wall[1], WALL_SIZE, WALL_SIZE), 1)
                # 绘制坦克
                self.player_tank.draw()
                for enemy in self.enemy_tanks:
                    enemy.draw()
                # 绘制子弹
                for bullet in self.bullets:
                    bullet.draw()
                # 绘制游戏信息面板
                self._draw_hud()

                # 更新屏幕显示
                pygame.display.flip()

            # 游戏结束后显示重新开始界面（带按钮）
            self._draw_game_over()

    def _draw_hud(self):
        """绘制计分、生命值、操作提示（更新为WASD）"""
        font = pygame.font.SysFont(None, 36)
        # 分数（左上角）
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        # 生命值（右上角，红色突出）
        lives_text = font.render(f"Lives: {self.player_tank.lives}", True, RED)
        screen.blit(lives_text, (SCREEN_WIDTH - 120, 10))
        # 操作提示（顶部中间，橙色醒目）：更新为WASD移动
        hint_text = font.render("鼠标瞄准 WASD移动 空格射击", True, ORANGE)
        screen.blit(hint_text, (SCREEN_WIDTH//2 - 200, 10))
        # 敌方提示（顶部偏右）
        enemy_hint = font.render("敌方：锁定瞄准+三发散射", True, BLUE)
        screen.blit(enemy_hint, (SCREEN_WIDTH//2 + 50, 45))
        # 子弹反弹提示
        bounce_hint = font.render("子弹：撞墙/边框均可反弹", True, YELLOW)
        screen.blit(bounce_hint, (10, 45))

    def _draw_game_over(self):
        """绘制游戏结束界面，支持点击重新开始按钮重启"""
        font_large = pygame.font.SysFont(None, 80)
        font_mid = pygame.font.SysFont(None, 48)
        
        # 游戏结束文字
        game_over_text = font_large.render("GAME OVER", True, RED)
        # 最终分数
        score_text = font_mid.render(f"Final Score: {self.score}", True, WHITE)
        
        # 居中计算坐标
        go_x = SCREEN_WIDTH//2 - game_over_text.get_width()//2
        go_y = SCREEN_HEIGHT//2 - 80
        s_x = SCREEN_WIDTH//2 - score_text.get_width()//2
        s_y = SCREEN_HEIGHT//2 - 20

        # 等待玩家点击按钮重新开始
        waiting = True
        while waiting:
            clock.tick(FPS)
            mouse_pos = pygame.mouse.get_pos()
            
            # 绘制界面
            screen.fill(BLACK)  # 清空屏幕
            screen.blit(game_over_text, (go_x, go_y))
            screen.blit(score_text, (s_x, s_y))
            # 绘制重新开始按钮并获取按钮矩形
            button_rect = self.draw_restart_button(mouse_pos)
            
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # 检测鼠标点击按钮
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(mouse_pos):
                        waiting = False  # 退出等待，重新开始
            
            pygame.display.flip()
        
        # 重置游戏状态，准备下一局
        self.reset_game()

# ====================== 启动游戏 ======================
if __name__ == "__main__":
    game = TankGame()
    game.run()