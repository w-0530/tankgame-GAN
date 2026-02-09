#!/usr/bin/env python3
"""
坦克游戏AI观看脚本 - 简化版
观看游戏内置AI玩游戏，无需额外依赖
"""

import tankgame
import time
import argparse

def print_game_info():
    """打印游戏控制信息"""
    print("=" * 60)
    print("🎮 坦克游戏AI观看模式")
    print("=" * 60)
    print("控制说明:")
    print("  - 空格键: 暂停/继续")
    print("  - R键: 重新开始")
    print("  - Q键: 退出")
    print("  - A键: 启用/禁用AI自动开火")
    print()
    print("游戏信息:")
    print("  - 绿色坦克: AI控制的玩家")
    print("  - 蓝色坦克: 敌方AI")
    print("  - 黄色圆点: 玩家子弹")
    print("  - 红色圆点: 敌方子弹")
    print("=" * 60)
    print()

def simple_ai_agent(game):
    """简单的AI代理，基于游戏状态做决策"""
    state = game.get_state()
    
    # 获取玩家和最近敌人的信息
    player = game.player
    enemy = game._get_nearest_enemy()
    
    if not enemy or not enemy.alive:
        # 没有敌人时随机移动
        import random
        actions = [random.choice([tankgame.ACTION_UP, tankgame.ACTION_DOWN, 
                                tankgame.ACTION_LEFT, tankgame.ACTION_RIGHT])]
        if random.random() < 0.1:
            actions.append(random.choice([tankgame.ACTION_GUN_LEFT, tankgame.ACTION_GUN_RIGHT]))
        return actions
    
    # 计算到敌人的距离和角度
    dx = enemy.x - player.x
    dy = enemy.y - player.y
    distance = tankgame.distance_between(player.x, player.y, enemy.x, enemy.y)
    target_angle = tankgame.math.atan2(-dy, dx) % (2 * tankgame.math.pi)
    
    # 瞄准敌人
    actions = []
    angle_diff = target_angle - player.aim_angle
    
    # 标准化角度差到 [-π, π]
    while angle_diff > tankgame.math.pi:
        angle_diff -= 2 * tankgame.math.pi
    while angle_diff < -tankgame.math.pi:
        angle_diff += 2 * tankgame.math.pi
    
    # 根据角度差调整炮管
    if abs(angle_diff) > tankgame.math.pi / 18:  # 10度
        if angle_diff > 0:
            actions.append(tankgame.ACTION_GUN_LEFT)
        else:
            actions.append(tankgame.ACTION_GUN_RIGHT)
    
    # 移动策略
    if distance > 300:  # 太远了，接近敌人
        if abs(dx) > abs(dy):
            actions.append(tankgame.ACTION_LEFT if dx < 0 else tankgame.ACTION_RIGHT)
        else:
            actions.append(tankgame.ACTION_UP if dy < 0 else tankgame.ACTION_DOWN)
    elif distance < 150:  # 太近了，远离敌人
        if abs(dx) > abs(dy):
            actions.append(tankgame.ACTION_RIGHT if dx < 0 else tankgame.ACTION_LEFT)
        else:
            actions.append(tankgame.ACTION_DOWN if dy < 0 else tankgame.ACTION_UP)
    else:  # 理想距离，横向移动躲避
        if tankgame.random.random() < 0.7:
            actions.append(tankgame.random.choice([tankgame.ACTION_LEFT, tankgame.ACTION_RIGHT]))
    
    # 射击决策
    if abs(angle_diff) < tankgame.math.pi / 12:  # 15度内可以射击
        actions.append(tankgame.ACTION_SHOOT)
    
    return actions

def watch_ai_play(args):
    """观看AI玩游戏的主函数"""
    
    # 打印游戏信息
    print_game_info()
    
    # 创建游戏
    game = tankgame.TankGame(render=True)
    
    # 游戏统计
    episode_count = 0
    total_score = 0
    best_score = 0
    
    # 控制变量
    paused = False
    auto_restart = args.auto_restart
    ai_shoot_enabled = False
    
    print(f"🎯 开始观看AI游戏...")
    print(f"   自动重启: {'开启' if auto_restart else '关闭'}")
    print(f"   AI自动开火: {'开启' if ai_shoot_enabled else '关闭'} (按A键切换)")
    print()
    
    try:
        while True:
            episode_count += 1
            state = game.reset()
            step_count = 0
            
            # 启用/禁用AI自动开火
            if ai_shoot_enabled:
                game.enable_auto_shoot()
            else:
                game.disable_auto_shoot()
            
            print(f"📍 第 {episode_count} 局开始")
            
            # 游戏主循环
            while True:
                # 处理事件
                for event in tankgame.pygame.event.get():
                    if event.type == tankgame.pygame.QUIT:
                        print("👋 退出观看")
                        return
                    elif event.type == tankgame.pygame.KEYDOWN:
                        if event.key == tankgame.pygame.K_q:
                            print("👋 退出观看")
                            return
                        elif event.key == tankgame.pygame.K_SPACE:
                            paused = not paused
                            print(f"⏸️  {'暂停' if paused else '继续'}")
                        elif event.key == tankgame.pygame.K_r:
                            print("🔄 手动重新开始")
                            game.reset()
                            state = game.get_state()
                        elif event.key == tankgame.pygame.K_a:
                            ai_shoot_enabled = not ai_shoot_enabled
                            if ai_shoot_enabled:
                                game.enable_auto_shoot()
                                print("🔥 AI自动开火: 开启")
                            else:
                                game.disable_auto_shoot()
                                print("🔥 AI自动开火: 关闭")
                
                if paused:
                    tankgame.pygame.time.wait(100)
                    continue
                
                # AI决策
                if not ai_shoot_enabled:
                    actions = simple_ai_agent(game)
                    game.do_actions(actions)
                
                # 更新游戏
                reward, done = game.step()
                step_count += 1
                
                # 每60帧显示一次状态
                if step_count % 60 == 0:
                    player = game.player
                    enemy = game._get_nearest_enemy()
                    if enemy and enemy.alive:
                        distance = tankgame.distance_between(player.x, player.y, enemy.x, enemy.y)
                        print(f"🎯 第{episode_count}局 - 步数:{step_count:4d} | "
                              f"分数:{game.score:3d} | 生命:{player.lives} | "
                              f"距离:{distance:6.1f} | 时间:{game.remaining_time:2d}s")
                
                # 游戏结束处理
                if done:
                    total_score += game.score
                    if game.score > best_score:
                        best_score = game.score
                    
                    print(f"🏁 第 {episode_count} 局结束")
                    print(f"   游戏分数: {game.score}")
                    print(f"   总步数: {step_count}")
                    print(f"   剩余时间: {game.remaining_time}秒")
                    print(f"   历史最佳: {best_score}")
                    print(f"   平均分数: {total_score/episode_count:.1f}")
                    print()
                    
                    # 自动重启或等待
                    if auto_restart:
                        tankgame.pygame.time.wait(2000)  # 等待2秒
                        break
                    else:
                        print("按R键重新开始，Q键退出")
                        # 等待用户输入
                        waiting = True
                        while waiting:
                            for event in tankgame.pygame.event.get():
                                if event.type == tankgame.pygame.QUIT:
                                    return
                                elif event.type == tankgame.pygame.KEYDOWN:
                                    if event.key == tankgame.pygame.K_q:
                                        return
                                    elif event.key == tankgame.pygame.K_r:
                                        waiting = False
                                        break
                            tankgame.pygame.time.wait(100)
                        break
    
    except KeyboardInterrupt:
        print("\n👋 观看被中断")
    
    # 最终统计
    print("=" * 60)
    print("📊 观看统计:")
    print(f"   总局数: {episode_count}")
    print(f"   平均分数: {total_score/episode_count:.1f}")
    print(f"   最佳分数: {best_score}")
    print("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="观看AI玩坦克游戏")
    parser.add_argument("--auto-restart", action="store_true",
                       help="游戏结束后自动重新开始")
    
    args = parser.parse_args()
    
    # 开始观看
    watch_ai_play(args)

if __name__ == "__main__":
    main()