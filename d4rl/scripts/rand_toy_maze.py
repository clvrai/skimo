import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d
from d4rl.pointmaze.maze_layouts import sample_layout

# SIZE = 40
#
# RENDER_SCALE = 10
#
# maze_layout = np.zeros((SIZE, SIZE))

### Rand Fill Algorithm

# def fill_block(x, y, maze_layout):
#     if maze_layout[max(x-1, 0), y] or maze_layout[min(x+1, SIZE-1), y] or \
#         maze_layout[x, max(y-1, 0)] or maze_layout[x, min(y+1, SIZE-1)]:
#         return np.random.rand() < 0.5
#     return np.random.rand() < 0.1
#
# for x in range(SIZE):
#     for y in range(SIZE):
#         maze_layout[x, y] = 1 if fill_block(x, y, maze_layout) else 0


# ### Rand Wall-Place Algorithm
# MAX_LEN_FRAC = 0.2      # fraction of total maze size for max wall length
# COVERAGE_FRAC = 0.3     # until what fraction of maze covered we keep adding walls
# def place_wall(maze_layout):
#     sample_vert_hor = 0 if np.random.rand() < 0.5 else 1
#     sample_len = int(max(MAX_LEN_FRAC * SIZE * np.random.rand(), 3))
#     sample_pos1 = int(np.random.rand() * (SIZE - 3 - 1))
#     sample_pos2 = int(np.random.rand() * (SIZE - 1))
#     if sample_vert_hor == 0:
#         maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
#     else:
#         maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
#     return maze_layout
#
# while np.mean(maze_layout) < COVERAGE_FRAC:
#     maze_layout = place_wall(maze_layout)


# ### Rand Wall-Place Algorithm w/ doors
# MAX_LEN_FRAC = 0.5      # fraction of total maze size for max wall length
# MIN_LEN_FRAC = 0.3
# COVERAGE_FRAC = 0.4     # until what fraction of maze covered we keep adding walls
# def place_wall(maze_layout):
#     sample_vert_hor = 0 if np.random.rand() < 0.5 else 1
#     sample_len = int(max((MAX_LEN_FRAC-MIN_LEN_FRAC) * SIZE * np.random.rand() + MIN_LEN_FRAC*SIZE, 3))
#     sample_door_offset = np.random.choice(np.arange(1, sample_len - 1))
#     sample_pos1 = int(np.random.rand() * (SIZE - sample_len - 1))
#     sample_pos2 = int(np.random.rand() * (SIZE - 2) + 1)
#
#     if sample_vert_hor == 0:
#         maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
#         maze_layout[sample_pos1 + sample_door_offset, sample_pos2] = 0
#         maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 + 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 - 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 + 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 - 1] = 1
#     else:
#         maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
#         maze_layout[sample_pos2, sample_pos1 + sample_door_offset] = 0
#         maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset - 1] = 1
#         maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset - 1] = 1
#         maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset + 1] = 1
#         maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset + 1] = 1
#     return maze_layout
#
# while np.mean(maze_layout) < COVERAGE_FRAC:
#     maze_layout = place_wall(maze_layout)


# ### Rand Wall-Place Algorithm w/ doors and diversified sampling
# MAX_LEN_FRAC = 0.5      # fraction of total maze size for max wall length
# MIN_LEN_FRAC = 0.3
# COVERAGE_FRAC = 0.25     # until what fraction of maze covered we keep adding walls
#
#
# def compute_sampling_probs(maze_layout, axis, filter=None):
#     if filter is None:
#         filter = [1/5, 1/5, 1/5, 1/5, 1/5]
#     coverage = np.mean(maze_layout, axis=axis)
#     probs = np.convolve(coverage, np.array(filter), 'valid')
#     return np.exp(1 - probs) / np.sum(np.exp(1 - probs))
#
#
# def place_wall(maze_layout):
#     sample_vert_hor = 0 if np.random.rand() < 0.5 else 1
#     sample_len = int(max((MAX_LEN_FRAC-MIN_LEN_FRAC) * SIZE * np.random.rand() + MIN_LEN_FRAC*SIZE, 3))
#     sample_door_offset = np.random.choice(np.arange(1, sample_len - 1))
#     sample_pos1 = int(np.random.rand() * (SIZE - sample_len - 1))
#
#     # sample_pos2 = int(np.random.rand() * (SIZE - 2) + 1)
#
#     if sample_vert_hor == 0:
#         sample_pos1 = np.random.choice(np.arange(0, SIZE - sample_len+1), p=compute_sampling_probs(
#             maze_layout, axis=1, filter=np.ones(sample_len)/sample_len))
#         sample_pos2 = np.random.choice(np.arange(2, SIZE-2),
#                                        p=compute_sampling_probs(maze_layout[sample_pos1 : sample_pos1 + sample_len, :], axis=0))
#         maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
#         maze_layout[sample_pos1 + sample_door_offset, sample_pos2] = 0
#         maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 + 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 - 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 + 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 - 1] = 1
#     else:
#         sample_pos1 = np.random.choice(np.arange(0, SIZE - sample_len + 1), p=compute_sampling_probs(
#             maze_layout, axis=0, filter=np.ones(sample_len) / sample_len))
#         sample_pos2 = np.random.choice(np.arange(2, SIZE - 2),
#                                        p=compute_sampling_probs(maze_layout[:, sample_pos1: sample_pos1 + sample_len], axis=1))
#         maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
#         maze_layout[sample_pos2, sample_pos1 + sample_door_offset] = 0
#         maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset - 1] = 1
#         maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset - 1] = 1
#         maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset + 1] = 1
#         maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset + 1] = 1
#     return maze_layout
#
# while np.mean(maze_layout) < COVERAGE_FRAC:
#     maze_layout = place_wall(maze_layout)


### Rand Wall-Place Algorithm w/ doors and diversified sampling in 2D
# MAX_LEN_FRAC = 0.5      # fraction of total maze size for max wall length
# MIN_LEN_FRAC = 0.3
# COVERAGE_FRAC = 0.25     # until what fraction of maze covered we keep adding walls
# TEMP = 20


maze_layout = sample_layout(seed=0)


RENDER_SCALE = 10
render_maze_layout = maze_layout.repeat(RENDER_SCALE, axis=0).repeat(
    RENDER_SCALE, axis=1
)
render_maze_layout = np.flip(np.flip(render_maze_layout, 0), 1)

cv2.imwrite("toy_maze.png", (1 - render_maze_layout) * 255)
