import argparse
import d4rl
import gym
import tqdm
import matplotlib.pyplot as plt

from d4rl.pointmaze.maze_model import MazeEnv
from d4rl.pointmaze.semantic_maze_layouts import (
    SEMANTIC_MAZE_1_LAYOUT,
    SEMANTIC_MAZE_2_LAYOUT,
    semantic_layout2str,
    xy2id,
)


START_POS = [27.0, 4.0]
TARGET_POS = [18.0, 8.0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="kitchen-mixed-v0")
    args = parser.parse_args()

    env = MazeEnv(semantic_layout2str(SEMANTIC_MAZE_2_LAYOUT), agent_centric_view=False)
    env.reset()

    x, y = [], []
    for t in range(2000):
        obs, _, done, _ = env.step(env.action_space.sample())
        x.append(obs[0])
        y.append(obs[1])
        print(xy2id(obs[0], obs[1], SEMANTIC_MAZE_2_LAYOUT))
        env.render(mode="human")
        if done:
            env.reset()

    plt.scatter(x, y)
    plt.show()
