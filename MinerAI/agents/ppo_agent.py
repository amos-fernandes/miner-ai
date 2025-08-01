# Agente PPO com Stable-Baselines3
# agents/ppo_agent.py
"""
Agente PPO (Proximal Policy Optimization) para o jogo Miner.
Baseado em Stable-Baselines3.
"""
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.miner_env import MinerEnv

def train_ppo_agent(
    num_mines=3,
    total_timesteps=100_000,
    save_path="models/saved/ppo_miner",
    log_dir="logs/tensorboard"
):
    """
    Treina um agente PPO no ambiente MinerEnv.
    """
    # Criar ambiente
    env = MinerEnv(num_mines=num_mines)
    env = DummyVecEnv([lambda: env])  # Necessário para SB3

    # Criar modelo PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
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
        name_prefix="ppo_miner"
    )

    print(f"🚀 Iniciando treinamento do agente PPO ({num_mines} minas)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="PPO"
    )

    # Salvar modelo final
    model.save(save_path)
    print(f"✅ Modelo PPO salvo em: {save_path}.zip")
    return model


def load_ppo_agent(model_path="models/saved/ppo_miner"):
    """
    Carrega um modelo PPO treinado.
    """
    model = PPO.load(model_path)
    print(f"📥 Modelo PPO carregado de: {model_path}")
    return model


# Exemplo de uso
if __name__ == "__main__":
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)

    train_ppo_agent(num_mines=3, total_timesteps=100_000)