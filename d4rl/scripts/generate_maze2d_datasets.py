import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model, maze_layouts
import numpy as np
import pickle
import gzip
import h5py
import os
import argparse


def reset_data():
    return {
        "observations": [],
        "actions": [],
        "terminals": [],
        "rewards": [],
        "infos/goal": [],
        "infos/qpos": [],
        "infos/qvel": [],
    }


def append_data(data, s, a, tgt, done, env_data):
    data["observations"].append(s)
    data["actions"].append(a)
    data["rewards"].append(0.0)
    data["terminals"].append(done)
    data["infos/goal"].append(tgt)
    data["infos/qpos"].append(env_data.qpos.ravel().copy())
    data["infos/qvel"].append(env_data.qvel.ravel().copy())


def npify(data):
    for k in data:
        if k == "terminals":
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render trajectories")
    parser.add_argument("--noisy", action="store_true", help="Noisy actions")
    parser.add_argument(
        "--env_name", type=str, default="maze2d-umaze-v1", help="Maze type"
    )
    parser.add_argument(
        "--num_samples", type=int, default=int(1e6), help="Num samples to collect"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Base directory for dataset"
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=int(-1),
        help="(Optional) Index of generated data batch",
    )
    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)

    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = 0
    for _ in range(args.num_samples):
        position = s[0:2]
        velocity = s[2:4]
        act, done = controller.get_action(position, velocity, env._target)
        if args.noisy:
            act = act + np.random.randn(*act.shape) * 0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True
        append_data(data, s, act, env._target, done, env.sim.data)

        ns, _, _, _ = env.step(act)

        if len(data["observations"]) % 1000 == 0:
            print(len(data["observations"]))

        ts += 1
        if done:
            env.set_target()
            done = False
            ts = 0
        else:
            s = ns

        if args.render:
            env.render()

    if args.batch_idx >= 0:
        dir_name = (
            "maze2d-%s-noisy" % args.maze
            if args.noisy
            else "maze2d-%s-sparse" % args.maze
        )
        os.makedirs(os.path.join(args.data_dir, dir_name), exist_ok=True)
        fname = os.path.join(
            args.data_dir, dir_name, "rollouts_batch_{}.h5".format(args.batch_idx)
        )
    else:
        os.makedirs(args.data_dir, exist_ok=True)
        fname = (
            "maze2d-%s-noisy.hdf5" % args.maze
            if args.noisy
            else "maze2d-%s-sparse.hdf5" % args.maze
        )
        fname = os.path.join(args.data_dir, fname)

    dataset = h5py.File(fname, "w")
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression="gzip")


if __name__ == "__main__":
    main()
