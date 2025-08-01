
# evaluate.py
from stable_baselines3 import PPO
import gym

env = MinerEnv(grid_size=5, num_mines=3)
model = PPO.load("models/saved/miner_ppo")

obs = env.reset()
total_reward = 0

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        print(f"Rodada finalizada com multiplicador: {reward:.2f}")
        obs = env.reset()

print(f"Recompensa média: {total_reward / 100:.2f}")