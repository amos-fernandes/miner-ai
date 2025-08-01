# corrigir_minerai.py
import os
import shutil
import subprocess
import sys

# ===========================================
# SCRIPT DE CORREÇÃO AUTOMÁTICA PARA MINERAI
# Converte gym → gymnasium e corrige API
# ===========================================

def instalar_dependencias():
    """Instala gymnasium e shimmy"""
    print("📦 Instalando dependências necessárias: gymnasium e shimmy...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "gymnasium", "shimmy"
        ])
        print("✅ Dependências instaladas com sucesso!\n")
    except Exception as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        sys.exit(1)

def atualizar_miner_env():
    """Atualiza o arquivo envs/miner_env.py para gymnasium"""
    caminho ="MinerAI/envs/miner_env.py"
    if not os.path.exists(caminho):
        print(f"❌ Arquivo {caminho} não encontrado. Verifique a estrutura do projeto.")
        sys.exit(1)

    print(f"🔧 Atualizando {caminho} para gymnasium...")

    # Backup
    backup = caminho.replace(".py", "_backup.py")
    shutil.copy(caminho, backup)
    print(f"📁 Backup salvo como: {backup}")

    # Novo código compatível com gymnasium
    novo_codigo = '''
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MinerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=5, num_mines=3, render_mode=None):
        super(MinerEnv, self).__init__()
        self.grid_size = grid_size
        self.num_tiles = grid_size ** 2
        self.num_mines = num_mines
        self.render_mode = render_mode

        # Espaço de ações: 0 a 24 = tiles, 25 = SACAR
        self.action_space = spaces.Discrete(self.num_tiles + 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Estado interno
        self.board = None
        self.revealed = None
        self.tiles_clicked = 0
        self.multiplier = 1.0
        self.done = False

    def reset(self, seed=None, options=None):
        """Redefine o ambiente. Usa seed para reprodutibilidade."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Inicializa tabuleiro
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        mines_flat = self.np_random.choice(self.num_tiles, self.num_mines, replace=False)
        self.board.flat[mines_flat] = 1

        self.revealed = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.tiles_clicked = 0
        self.multiplier = 1.0
        self.done = False

        info = {"episode": {"length": 0, "reward": 0}}
        return self._get_state(), info

    def _get_state(self):
        remaining = self.num_tiles - self.tiles_clicked
        risk = self.num_mines / (remaining + 1e-8)
        return np.array([
            self.tiles_clicked / self.num_tiles,
            risk,
            min(self.multiplier / 10.0, 1.0)
        ], dtype=np.float32)

    def step(self, action):
        """Executa uma ação e retorna novo estado com nova API."""
        if action == self.num_tiles:  # SACAR
            reward = self.multiplier
            terminated = True
            truncated = False
            info = {"final_reward": reward}
            return self._get_state(), reward, terminated, truncated, info

        row, col = divmod(action, self.grid_size)
        if self.revealed[row, col]:
            return self._get_state(), -0.05, False, False, {}

        self.revealed[row, col] = True

        if self.board[row, col] == 1:
            terminated = True
            truncated = False
            reward = -self.multiplier
            info = {"final_reward": reward, "reason": "hit_mine"}
            return self._get_state(), reward, terminated, truncated, info
        else:
            self.tiles_clicked += 1
            self.multiplier = 1.0 + 0.3 * self.tiles_clicked
            return self._get_state(), 0.1, False, False, {}

    def render(self, mode="human"):
        if self.render_mode == "human":
            print("\\nBoard:")
            display = np.where(self.revealed, self.board, -1)
            print(display)
'''

    # Salvar novo código
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(novo_codigo.strip() + "\\n")
    print(f"✅ {caminho} atualizado com sucesso para gymnasium!\\n")

def testar_ambiente():
    """Testa o ambiente corrigido"""
    print("🧪 Testando o ambiente MinerEnv...")
    try:
        from envs.miner_env import MinerEnv
        env = MinerEnv(num_mines=3)
        obs, info = env.reset(seed=42)
        print(f"✅ reset() OK - Observação: {obs}")

        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✅ step() {i+1}: ação={action}, recompensa={reward}, finalizado={terminated or truncated}")
            if terminated or truncated:
                break

        env.close()
        print("✅ Ambiente MinerEnv está funcionando corretamente!\\n")
    except Exception as e:
        print(f"❌ Erro ao testar o ambiente: {e}")
        sys.exit(1)

def atualizar_train_py():
    """Garante que train.py está correto"""
    caminho = "train.py"
    if not os.path.exists(caminho):
        print(f"⚠️ {caminho} não encontrado. Crie o arquivo com o conteúdo padrão.")
        return

    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()

    # Garante que não há import errado
    if "import gym" in conteudo:
        print(f"⚠️ Detectado 'import gym' em {caminho}. Corrigindo...")
        conteudo = conteudo.replace("import gym", "import gymnasium as gym")  # só se usar
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
        print(f"✅ {caminho} atualizado.")

    print(f"✅ {caminho} verificado.\\n")

# ============
# EXECUÇÃO
# ============

if __name__ == "__main__":
    print("🚀 Iniciando correção automática do MinerAI\\n")
    
    # Passo 1: Instalar dependências
    #instalar_dependencias()
    
    # Passo 2: Atualizar miner_env.py
    atualizar_miner_env()
    
    # Passo 3: Verificar train.py
    atualizar_train_py()
    
    # Passo 4: Testar ambiente
    testar_ambiente()
    
    print("🎉 \\033[1mCorreção concluída com sucesso!\\033[0m")
    print("\\n▶️ Agora você pode executar:")
    print("   python train.py")
    print("\\n💡 Dica: Use `tensorboard --logdir=logs/tensorboard` para acompanhar o treinamento.")