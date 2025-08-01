
# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.miner_env import MinerEnv
import os

if __name__ == "__main__":
    env = MinerEnv(grid_size=5, num_mines=3)
    env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/tensorboard/")
    
    print("Treinando o agente...")
    model.learn(total_timesteps=100_000)

    model.save("models/saved/miner_ppo")
    print("Treinamento concluído! Modelo salvo.")