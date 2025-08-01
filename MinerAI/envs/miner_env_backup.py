# TODO: Implementar o ambiente Gym para o jogo Miner
# envs/miner_env.py
import gym
from gym import spaces
import numpy as np

class MinerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5, num_mines=3):
        super(MinerEnv, self).__init__()
        self.grid_size = grid_size
        self.num_tiles = grid_size ** 2
        self.num_mines = num_mines

        self.action_space = spaces.Discrete(self.num_tiles + 1)  # +1 para SACAR
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        mines_flat = np.random.choice(self.num_tiles, self.num_mines, replace=False)
        self.board.flat[mines_flat] = 1
        self.revealed = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.tiles_clicked = 0
        self.multiplier = 1.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        remaining_tiles = self.num_tiles - self.tiles_clicked
        risk = self.num_mines / (remaining_tiles + 1e-8)
        return np.array([
            self.tiles_clicked / self.num_tiles,
            risk,
            min(self.multiplier / 10.0, 1.0)
        ], dtype=np.float32)

    def step(self, action):
        if action == self.num_tiles:  # SACAR
            reward = self.multiplier
            self.done = True
            return self._get_state(), reward, self.done, {}

        row, col = divmod(action, self.grid_size)
        if self.revealed[row, col]:
            return self._get_state(), -0.05, False, {}

        self.revealed[row, col] = True

        if self.board[row, col] == 1:
            self.done = True
            return self._get_state(), -self.multiplier, self.done, {}
        else:
            self.tiles_clicked += 1
            self.multiplier = 1.0 + 0.3 * self.tiles_clicked  # ex: 1.3, 1.6, ...
            return self._get_state(), 0.1, self.done, {}

    def render(self, mode='human'):
        print("Board:")
        display = np.where(self.revealed, self.board, -1)
        print(display)