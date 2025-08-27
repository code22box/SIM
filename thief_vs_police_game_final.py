####Run & Escape: The Infinite Maze Game

import numpy as np
import pygame
import random
import heapq
from openai import OpenAI
import re
import textwrap
import os
import requests
import datetime

# ダウンロードするファイル名を決定する
now = datetime.datetime.now()
image_filename = './images/image_' + now.strftime('%Y%m%d_%H%M%S') + '.png'

# 1文ごとに分割する関数
def split_sentences(text_list):
    sentences = []
    for text in text_list:
        # 正規表現で文末の句読点とその後のスペースを探し、分割
        sentences.extend(re.split(r'(?<=[.!?])\s+', text))
    return sentences

def split_sentences_with_linebreaks(sentences, max_length):
    result = []
    for sentence in sentences:
        wrapped = textwrap.wrap(sentence, width=max_length)
        result.extend(wrapped)
    return result

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text

# OpenAIのAPIを用いてテキストを生成する関数
def generate_text():
    try:
        client = OpenAI()
        prompt = """You are a creative game creator. You are responsible for generating text for use in the game.

        The game is about a player-controlled character running away from three enemy characters. There are many possible combinations of characters. For example, the player may be a thief and the enemy a police officer, the player may be a human and the enemy an alien, or the player may be a human and the enemy a zombie.
        
        You are then responsible for generating the game intro that will appear on the initial game screen, the game caption, and the messages that will appear during the play screen.
        
        Now, feel free to come up with your own settings, and generate only one set of the following necessary information. Never output any information other than Intro, Caption, and Message.
        
        Intro:Welcome to 'Thief vs Police'! You are a cunning thief trying to escape from the police. Use your wits and agility to escape the police. Good luck!
        Caption:Thief vs Police
        Message:Run away from the police!
              
        Intro:Welcome to 'Human vs Aliens'! You are a brave human fighting against an alien invasion. Use your quick thinking and agility to avoid capture and save humanity. Good luck!
        Caption:Human vs Aliens
        Message:Dodge the aliens!
        
        Intro:
        Caption:
        Message:
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative game creator."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        result = response.choices[0].message.content.split('\n')
        
        # 各行が "Intro:" で始まる場合、その行を抽出し改行を挿入する
        intro_lines = [line for line in result if line.startswith("Intro:")]
        intro_lines = [remove_prefix(line, "Intro:") for line in intro_lines]
        intro_text = split_sentences_with_linebreaks(split_sentences(intro_lines), 45)
        
        # 各行が "Caption:" で始まる場合、その行を連結する
        caption_lines = [line for line in result if line.startswith("Caption:")]
        caption_lines = [remove_prefix(line, "Caption:") for line in caption_lines]
        caption = "".join(caption_lines)

        # 各行が "Message:" で始まる場合、その行を連結する
        message_lines = [line for line in result if line.startswith("Message:")]
        message_lines = [remove_prefix(line, "Message:") for line in message_lines]
        message = "".join(message_lines)
        
        if not intro_text or not caption or not message:
            raise ValueError("Incomplete response from API")

    except Exception:
        # APIが失敗した場合のデフォルトのテキスト
        intro_text = split_sentences_with_linebreaks(split_sentences(["Welcome to 'Thief vs Police'! You are a cunning thief trying to escape from the police. Use your wits and agility to escape the police. Good luck!"]), 45)
        caption = "".join(["Thief vs Police"])
        message = "".join(["Run away from the police!"])

    return intro_text, caption, message

# テキストを取得
intro_text, game_caption, game_message = generate_text()

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt=f"{intro_text}, {game_caption}, {game_message}, best quality, 4K",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url

print(image_url)

# 画像をダウンロードして保存する
response = requests.get(image_url)
if response.status_code == 200:
    with open(image_filename, 'wb') as f:
        f.write(response.content)
    print(f"Image saved as {image_filename}")

# フィールドの生成関数
def generate_percolation_field(size, p):
    return np.random.rand(size, size) < p

# ヒューリスティック関数
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A*アルゴリズム関数
def a_star(start, goal, field):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < size:
                if 0 <= neighbor[1] < size:                
                    if not field[neighbor[0]][neighbor[1]]:
                        continue
                else:
                    continue
            else:
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return False

# エージェントクラスの定義
class Agent:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def move(self, dx, dy):
        new_x = max(0, min(size - 1, self.x + dx))
        new_y = max(0, min(size - 1, self.y + dy))
        if field[new_x, new_y]:  # 移動先が白色（True）なら移動を許可
            self.x = new_x
            self.y = new_y

# A*アルゴリズムを使うエージェントクラスの定義
class AStarAgent(Agent):
    def move(self, thief, other_cops):
        path = a_star((self.x, self.y), (thief.x, thief.y), field)
        if path and len(path) > 1:
            next_move = path[1]
            dx = next_move[0] - self.x
            dy = next_move[1] - self.y

            # 他の警察との衝突を避ける
            if any(cop.x == self.x + dx and cop.y == self.y + dy for cop in other_cops):
                for alternative_action in actions:
                    alt_x = self.x + alternative_action[0]
                    alt_y = self.y + alternative_action[1]
                    if not any(cop.x == alt_x and cop.y == alt_y for cop in other_cops):
                        dx, dy = alternative_action
                        break

            super().move(dx, dy)

# フィールドを描画する関数
def draw_field(screen, field):
    for x in range(size):
        for y in range(size):
            color = WHITE if field[x, y] else BLACK
            pygame.draw.rect(screen, color, (x * cell_size, y * cell_size, cell_size, cell_size))

# エージェントを描画する関数
def draw_agent(screen, agent):
    pygame.draw.rect(screen, agent.color, (agent.x * cell_size, agent.y * cell_size, cell_size, cell_size))

# 情報を描画する関数
def draw_info(screen, play_time, message):
    font = pygame.font.Font(None, 36)
    text_surface = font.render(f"Time: {play_time:.1f}s", True, WHITE)
    screen.blit(text_surface, (screen_width - 400, 50))
    
    message_surface = font.render(message, True, WHITE)
    screen.blit(message_surface, (screen_width - 450, 100))

# 空きセルを見つける関数
def find_empty_cell(field):
    size = len(field)
    while True:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if field[x, y]:
            return x, y

# 初期画面を表示する関数
def show_intro_screen(screen, intro_text):
    intro_duration = 20 * 1000  # 1秒
    intro_start_time = pygame.time.get_ticks()
    image = pygame.image.load(image_filename)  # 右側に表示する画像
    image = pygame.transform.scale(image, (600, 600))  # 画像を右半分に収める
    font = pygame.font.Font(None, 36)

    while pygame.time.get_ticks() - intro_start_time < intro_duration:
        screen.fill(BLACK)
        screen.blit(image, (screen_width // 2, 0))

        elapsed_time = pygame.time.get_ticks() - intro_start_time
        text_start_y = screen_height - (elapsed_time // 30) % screen_height

        for i, line in enumerate(intro_text):
            text_surface = font.render(line, True, WHITE)
            screen.blit(text_surface, (50, text_start_y + i * 40))

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
    return True

# カウントダウンを表示する関数
def show_countdown(screen):
    countdown_duration = 5 * 1000  # 5秒
    countdown_start_time = pygame.time.get_ticks()
    font = pygame.font.Font(None, 74)

    while pygame.time.get_ticks() - countdown_start_time < countdown_duration:
        screen.fill(BLACK)
        draw_field(screen, field)
        draw_agent(screen, thief)
        for cop in cops:
            draw_agent(screen, cop)

        elapsed_time = pygame.time.get_ticks() - countdown_start_time
        countdown_value = 5 - elapsed_time // 1000

        if countdown_value > 0:
            countdown_surface = font.render(str(countdown_value), True, RED)
        else:
            countdown_surface = font.render("Start", True, RED)
        
        screen.blit(countdown_surface, (screen_width // 2 + 400, screen_height // 2 - 50))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
    return True

size = 40  # フィールドのサイズ
p = 0.7  # パーコレーションの確率
field = generate_percolation_field(size, p)

# Pygameの初期化
pygame.init()

# 画面の設定
screen_width = 1300
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption(game_caption)
clock = pygame.time.Clock()

# 色の設定
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# セルのサイズ
cell_size = screen_height // size

# 状態空間と行動空間の定義
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 初期化
def initialize_game():
    global field, thief, cops, start_time, frame_count
    field = generate_percolation_field(size, p)
    thief_start = find_empty_cell(field)
    thief = Agent(thief_start[0], thief_start[1], BLUE)
    cops = [AStarAgent(*find_empty_cell(field), RED) for _ in range(3)]
    start_time = pygame.time.get_ticks()
    frame_count = 0

# 点滅アニメーションの関数
def blink_animation(screen, cops):
    for _ in range(15):  # 3秒間点滅（0.2秒間隔で点滅）
        screen.fill(BLACK)
        draw_field(screen, field)
        for cop in cops:
            draw_agent(screen, cop)
        pygame.display.flip()
        pygame.time.wait(100)
        screen.fill(BLACK)
        draw_field(screen, field)
        pygame.display.flip()
        pygame.time.wait(100)

# ゲームオーバー画面の表示
def game_over_screen(play_time):
    screen.fill(BLACK)
    font = pygame.font.Font(None, 74)
    game_over_surface = font.render("You have been caught!", True, WHITE)
    screen.blit(game_over_surface, (screen_width // 2 - 300, screen_height // 2 - 100))
    
    font = pygame.font.Font(None, 36)
    escape_time_surface = font.render(f"Survived Time: {play_time:.1f} seconds", True, WHITE)
    screen.blit(escape_time_surface, (screen_width // 2 - 200, screen_height // 2))
    
    button_rect = pygame.Rect(screen_width // 2 - 100, screen_height // 2 + 50, 200, 50)
    pygame.draw.rect(screen, WHITE, button_rect)
    font = pygame.font.Font(None, 36)
    button_surface = font.render("Play Again", True, BLACK)
    screen.blit(button_surface, (button_rect.x + 25, button_rect.y + 10))

    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    return True

initialize_game()

# メインループ
while True:
    if not show_intro_screen(screen, intro_text):
        break
    if not show_countdown(screen):
        break

    running = True
    while running:
        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    thief.move(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    thief.move(1, 0)
                elif event.key == pygame.K_UP:
                    thief.move(0, -1)
                elif event.key == pygame.K_DOWN:
                    thief.move(0, 1)
                    
        if not running:
            break
        
        # 操作キャラが敵に捕まるかどうかを確認
        if any(cop.x == thief.x and cop.y == thief.y for cop in cops):
            # 点滅アニメーションを再生
            blink_animation(screen, cops)
            
            # ゲームオーバー画面を表示
            if not game_over_screen(play_time):
                running = False
                break
            else:
                # ゲームを再初期化
                initialize_game()
                if not show_intro_screen(screen, intro_text):
                    running = False
                    break
                if not show_countdown(screen):
                    running = False
                    break
                continue

        # 現在のプレイ時間を計算
        current_time = pygame.time.get_ticks()
        play_time = (current_time - start_time) / 1000  # ミリ秒を秒に変換

        # 敵エージェントの移動処理（30フレームに1回）
        if frame_count % 30 == 0:
            for cop in cops:
                other_cops = [other_cop for other_cop in cops if other_cop != cop]
                cop.move(thief, other_cops)

        # 画面の描画処理
        screen.fill(BLACK)
        draw_field(screen, field)
        draw_agent(screen, thief)
        for cop in cops:
            draw_agent(screen, cop)
        
        # 情報の描画
        draw_info(screen, play_time, game_message)
        
        # 画面の更新
        pygame.display.flip()
        
        # フレームレートの調整
        clock.tick(30)  # フレームレートを30に設定
        frame_count += 1  # フレームカウントを増加

pygame.quit()
