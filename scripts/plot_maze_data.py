import pickle
import gzip

import matplotlib.pyplot as plt
import numpy as np
import imageio


print("Load pre-training data")
data = pickle.load(gzip.open("data/maze_states_skild.pkl.gz", "rb"))
data_size = len(data)
print(f"Load {data_size} trajectories")

maze_size = 40
extent = (0, maze_size, 0, maze_size)
overlay = imageio.imread(f"envs/assets/maze_overlay_{maze_size}.png")
plt.imshow(overlay, interpolation="none", alpha=0.3, extent=extent)
plt.xlim(0, maze_size)
plt.ylim(0, maze_size)
plt.axis("equal")


def render(data, cmap):
    plt.scatter(data[:, 0], data[:, 1], s=3, c=np.arange(len(data)), cmap=cmap)


for d in data:
    render(d["obs"], "cool")

plt.savefig("maze_data_plot.png")
