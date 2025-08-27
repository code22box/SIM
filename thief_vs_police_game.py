import pygame
import random
import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.envs import DummyVecEnv

# 初期化
pygame.init()

# 画面サイズ
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# 色
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)

# プレイヤー
player_size = 50
player_pos = [width // 2, height // 2]

# 警察
police_size = 50
police_pos = [random.randint(0, width-police_size), random.randint(0, height-police_size)]

# ゲームループ
game_over = False
clock = pygame.time.Clock()

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_pos[0] -= 5
    if keys[pygame.K_RIGHT]:
        player_pos[0] += 5
    if keys[pygame.K_UP]:
        player_pos[1] -= 5
    if keys[pygame.K_DOWN]:
        player_pos[1] += 5

    screen.fill(black)

    pygame.draw.rect(screen, blue, (player_pos[0], player_pos[1], player_size, player_size))
    pygame.draw.rect(screen, red, (police_pos[0], police_pos[1], police_size, police_size))

    pygame.display.flip()

    clock.tick(30)

pygame.quit()
