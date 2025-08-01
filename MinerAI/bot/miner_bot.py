# bot/miner_bot.py
import json
import time
import numpy as np
from stable_baselines3 import PPO
from envs.miner_env import MinerEnv
from vision.vision_detector import VisionDetector
from utils.human_behavior import human_delay
import pyautogui

class MinerBot:
    def __init__(self, model_path="models/saved/miner_ppo", config_path="config/config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        self.model = PPO.load(model_path)
        self.env = MinerEnv(num_mines=self.config["num_mines"])
        self.detector = VisionDetector(self.config["template_path"])
        self.state = self.env.reset()
        self.done = False
        self.tiles_clicked = 0

    def run(self):
        print("🔍 Procurando a grade do jogo...")
        region = self.config["screen_region"]
        tiles = self.detector.detect_grid(region)

        if len(tiles) != 25:
            print(f"⚠️ Detectados {len(tiles)} tiles. Esperado: 25.")
            return

        print(f"✅ Grade detectada com {len(tiles)} tiles.")

        for i, (tx, ty) in enumerate(tiles):
            if self.done:
                print("💥 Acertou mina! Parando.")
                break

            action, _ = self.model.predict(self.state, deterministic=True)
            if action == 25:  # SACAR
                print(f"💰 Agente decidiu SACAR após {self.tiles_clicked} cliques.")
                break

            screen_x = region["left"] + tx + self.detector.template_w // 2
            screen_y = region["top"] + ty + self.detector.template_h // 2

            pyautogui.moveTo(screen_x, screen_y, duration=np.random.uniform(0.3, 0.8))
            human_delay()
            pyautogui.click()
            human_delay()

            obs, reward, done, info = self.env.step(action)
            self.state = obs
            self.done = done
            self.tiles_clicked += 1
            print(f"🖱️ Clicado no tile {i+1} em ({screen_x}, {screen_y})")

        print("🏁 Bot finalizado.")