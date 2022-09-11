import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model, maze_layouts
import numpy as np
import pickle
import gzip
import h5py
import argparse
import os
import tqdm


START_POS = np.array([10.0, 24.0])
TARGET_POS = np.array([18.0, 8.0])


def reset_data():
    return {
        "states": [],
        "actions": [],
        "images": [],
        "terminals": [],
        "infos/goal": [],
        "infos/qpos": [],
        "infos/qvel": [],
    }


def append_data(data, s, a, img, tgt, done, env_data):
    data["states"].append(s)
    data["actions"].append(a)
    data["images"].append(img)
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


def sample_env_and_controller(args):
    layout_str = maze_layouts.rand_layout(seed=0, size=40)
    env = maze_model.MazeEnv(layout_str, agent_centric_view=args.agent_centric)
    controller = waypoint_controller.WaypointController(layout_str)
    return env, controller


def reset_env(env, agent_centric=False):
    s = env.reset()
    env.set_target(TARGET_POS)
    s = env.reset_to_location(START_POS)
    if agent_centric:
        [
            env.render(mode="rgb_array") for _ in range(100)
        ]  # so that camera can catch up with agent
    return s


def save_video(file_name, frames, fps=20, video_format="mp4"):
    import skvideo.io

    skvideo.io.vwrite(
        file_name,
        frames,
        inputdict={"-r": str(int(fps)),},
        outputdict={
            "-f": video_format,
            "-pix_fmt": "yuv420p",  # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render trajectories")
    parser.add_argument("--noisy", action="store_true", help="Noisy actions")
    parser.add_argument(
        "--agent_centric",
        action="store_true",
        help="Whether agent-centric images are rendered.",
    )
    parser.add_argument(
        "--save_images", action="store_true", help="Whether rendered images are saved."
    )
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Base directory for dataset"
    )
    parser.add_argument(
        "--num_samples", type=int, default=int(2e5), help="Num samples to collect"
    )
    parser.add_argument(
        "--min_traj_len",
        type=int,
        default=int(20),
        help="Min number of samples per trajectory",
    )
    parser.add_argument(
        "--rand_maze_size", type=int, default=int(20), help="Size of generate maze"
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=int(-1),
        help="(Optional) Index of generated data batch",
    )
    args = parser.parse_args()
    if args.agent_centric and not args.save_images:
        raise ValueError("Need to save images for agent-centric dataset")

    max_episode_steps = 1600  # if not args.save_images else 500
    env, controller = sample_env_and_controller(args)

    s = reset_env(env, agent_centric=args.agent_centric)

    data = reset_data()
    ts, cnt = 0, 0
    for tt in tqdm.tqdm(range(args.num_samples)):
        position = s[0:2]
        velocity = s[2:4]

        try:
            act, done = controller.get_action(position, velocity, env._target)
        except ValueError:
            # failed to find valid path to goal
            print("FAIL")
            data = reset_data()
            env, controller = sample_env_and_controller(args)
            s = reset_env(env, agent_centric=args.agent_centric)
            ts = 0
            continue

        print(act)

        if args.noisy:
            act = act + np.random.randn(*act.shape) * 0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True
        append_data(
            data,
            s,
            act,
            env.render(mode="rgb_array"),  # , camera_name='birdview'),
            env._target,
            done,
            env.sim.data,
        )

        ns, _, _, _ = env.step(act)

        ts += 1
        if done:
            if len(data["actions"]) > args.min_traj_len:
                save_data(args, data, cnt)
                print("Saved Demonstration. Exiting...")
                exit(0)
            data = reset_data()
            env, controller = sample_env_and_controller(args)
            s = reset_env(env, agent_centric=args.agent_centric)
            ts = 0
        else:
            s = ns

        if args.render:
            env.render(mode="human")


def save_data(args, data, idx):
    # save_video("seq_{}_ac.mp4".format(idx), data['images'])
    dir_name = ""
    if args.batch_idx >= 0:
        dir_name = os.path.join(dir_name, "batch_{}".format(args.batch_idx))
    os.makedirs(os.path.join(args.data_dir, dir_name), exist_ok=True)
    file_name = os.path.join(args.data_dir, dir_name, "rollout_{}.h5".format(idx))

    # save rollout to file
    f = h5py.File(file_name, "w")
    f.create_dataset("traj_per_file", data=1)

    # store trajectory info in traj0 group
    npify(data)
    traj_data = f.create_group("traj0")
    traj_data.create_dataset("states", data=data["states"])
    if args.save_images:
        traj_data.create_dataset("images", data=data["images"], dtype=np.uint8)
    else:
        traj_data.create_dataset(
            "images", data=np.zeros((data["states"].shape[0], 2, 2, 3), dtype=np.uint8)
        )
    traj_data.create_dataset("actions", data=data["actions"])

    terminals = data["terminals"]
    if np.sum(terminals) == 0:
        terminals[-1] = True

    # build pad-mask that indicates how long sequence is
    is_terminal_idxs = np.nonzero(terminals)[0]
    pad_mask = np.zeros((len(terminals),))
    pad_mask[: is_terminal_idxs[0]] = 1.0
    traj_data.create_dataset("pad_mask", data=pad_mask)

    f.close()


if __name__ == "__main__":
    main()
