
# simulate.py
import pygame
import numpy as np
from envs.miner_env import MinerEnv

pygame.init()
screen = pygame.display.set_mode((500, 600))
pygame.display.set_caption("MinerAI - Simulador")
font = pygame.font.SysFont(None, 36)

env = MinerEnv()
obs = env.reset()
clock = pygame.time.Clock()
running = True

colors = {
    "bg": (30, 30, 30),
    "tile": (100, 100, 100),
    "safe": (0, 200, 100),
    "mine": (255, 50, 50),
    "text": (255, 255, 255)
}

while running:
    screen.fill(colors["bg"])
    board = np.zeros((5, 5), dtype=str)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                obs = env.reset()

    # Renderizar tiles
    for r in range(5):
        for c in range(5):
            rect = pygame.Rect(c*90 + 10, r*90 + 10, 80, 80)
            if env.revealed[r][c]:
                color = colors["safe"] if env.board[r][c] == 0 else colors["mine"]
                pygame.draw.rect(screen, color, rect)
                text = font.render("💎" if env.board[r][c] == 0 else "💣", True, (0,0,0))
                screen.blit(text, rect.center)
            else:
                pygame.draw.rect(screen, colors["tile"], rect)
                text = font.render("?", True, colors["text"])
                screen.blit(text, rect.center)

    # Mostrar multiplicador
    mult_text = font.render(f"Mult: {env.multiplier:.2f}x", True, colors["text"])
    screen.blit(mult_text, (10, 470))

    status = "Vivo" if not env.done else "MORTO!"
    status_text = font.render(status, True, colors["text"])
    screen.blit(status_text, (10, 520))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()