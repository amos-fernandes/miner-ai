# Agente DQN com PyTorch
# agents/dqn_agent.py
"""
Agente DQN (Deep Q-Network) para o jogo Miner.
Baseado em PyTorch e Stable-Baselines3.
"""
import os
import torch
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from envs.miner_env import MinerEnv

# Custom Feature Extractor (opcional para melhor desempenho)
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=64)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, observations):
        return self.net(observations)

def train_dqn_agent(
    num_mines=3,
    total_timesteps=100_000,
    save_path="models/saved/dqn_miner",
    log_dir="logs/tensorboard"
):
    """
    Treina um agente DQN no ambiente MinerEnv.
    """
    env = MinerEnv(num_mines=num_mines)
    env = DummyVecEnv([lambda: env])

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=5e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        # Usar extractor personalizado
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            net_arch=[64, 64]
        )
    )

    # Callbacks
    eval_env = DummyVecEnv([lambda: MinerEnv(num_mines=num_mines)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}_best",
        log_path="logs/eval",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="./models/checkpoints/",
        name_prefix="dqn_miner"
    )

    print(f"🚀 Iniciando treinamento do agente DQN ({num_mines} minas)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="DQN"
    )

    model.save(save_path)
    print(f"✅ Modelo DQN salvo em: {save_path}.zip")
    return model


def load_dqn_agent(model_path="models/saved/dqn_miner"):
    """
    Carrega um modelo DQN treinado.
    """
    model = DQN.load(model_path)
    print(f"📥 Modelo DQN carregado de: {model_path}")
    return model


# Exemplo de uso
if __name__ == "__main__":
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)

    train_dqn_agent(num_mines=3, total_timesteps=100_000)