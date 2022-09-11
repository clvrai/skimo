import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
from d4rl.pointmaze.semantic_maze_layouts import (
    SEMANTIC_MAZE_LAYOUTS,
    SKILL2ID,
    semantic_layout2str,
    xy2id,
    xy2coord,
)
import numpy as np
import pickle
import gzip
import h5py
import argparse
import os
import tqdm


# Demos Semantic Maze 1
START_POS = np.array([5.0, 5.0])
TARGET_POS = np.array([37.0, 20.0])

# Demos Semantic Maze 2
# START_POS = np.array([36., 34.])
# TARGET_POS = np.array([3., 23.])


def reset_data():
    return {
        "states": [],
        "actions": [],
        "images": [],
        "terminals": [],
        "skills": [],
        "infos/goal": [],
        "infos/qpos": [],
        "infos/qvel": [],
    }


def append_data(data, s, a, img, tgt, done, skill, env_data):
    data["states"].append(s)
    data["actions"].append(a)
    data["images"].append(img)
    data["terminals"].append(done)
    data["skills"].append(skill)
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


def sample_env_and_controller(args, layout):
    layout_str = semantic_layout2str(layout)
    env = maze_model.MazeEnv(layout_str, agent_centric_view=args.agent_centric)
    controller = waypoint_controller.WaypointController(layout_str)
    return env, controller


def reset_env(env, agent_centric=False):
    s = env.reset()
    # env.set_target()
    env.set_target(TARGET_POS)
    s = env.reset_to_location(START_POS)
    if agent_centric:
        [
            env.render(mode="rgb_array") for _ in range(100)
        ]  # so that camera can catch up with agent
    return s


def get_skill(state, last_room_id, layout, skill_boundaries):
    x, y = state[0], state[1]
    xc, yc = xy2coord(x, y)

    # initialize room ID
    if last_room_id is None:
        last_room_id = xy2id(x, y, layout)
        if last_room_id == -1:
            # spawned on a doorway --> use any of the adjacent rooms
            last_room_id = np.random.choice(skill_boundaries[(xc, yc)])

    curr_room_id = xy2id(x, y, layout)

    if curr_room_id == 0:
        # this happens when agent is too close to wall and rounding pushes it into a wall cell --> keep last room value
        curr_room_id = last_room_id

    skill = None
    if curr_room_id != last_room_id and curr_room_id == -1:
        # entered a doorway
        connected_rooms = skill_boundaries[(xc, yc)]
        assert (
            last_room_id in connected_rooms
        )  # need to have come from one room adjacent to door
        skill = (
            str(connected_rooms[0]) + "-" + str(connected_rooms[1])
            if connected_rooms[0] == last_room_id
            else str(connected_rooms[1]) + "-" + str(connected_rooms[0])
        )

    return skill, curr_room_id


def fill_skills(data, skill):
    """Fill all trailing None with skill."""
    for i in reversed(range(len(data["skills"]))):
        if data["skills"][i] is None:
            data["skills"][i] = SKILL2ID[skill]
        else:
            break


def pad_skills(data):
    """Fill all trailing None with last non-None skill."""
    for i in reversed(range(len(data["skills"]))):
        if data["skills"][i] is not None:
            break
        for k in range(i, len(data["skills"])):
            data["skills"][k] = data["skills"][i]


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
        "--layout_id", type=int, default=int(-1), help="ID of layout used"
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

    layout, skill_boundaries = SEMANTIC_MAZE_LAYOUTS[args.layout_id]

    max_episode_steps = 1600 if not args.save_images else 500
    env, controller = sample_env_and_controller(args, layout)

    s = reset_env(env, agent_centric=args.agent_centric)

    data = reset_data()
    ts, cnt = 0, 0
    last_room_id = None
    for tt in tqdm.tqdm(range(args.num_samples)):
        position = s[0:2]
        velocity = s[2:4]

        try:
            act, done = controller.get_action(position, velocity, env._target)
        except ValueError:
            # failed to find valid path to goal
            data = reset_data()
            env, controller = sample_env_and_controller(args, layout)
            s = reset_env(env, agent_centric=args.agent_centric)
            ts = 0
            last_room_id = None
            continue

        if args.noisy:
            act = act + np.random.randn(*act.shape) * 0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True
        skill, last_room_id = get_skill(s, last_room_id, layout, skill_boundaries)
        if skill is not None:
            # fill all skills that were filled with None before
            fill_skills(data, skill)
        append_data(
            data,
            s,
            act,
            env.render(mode="rgb_array"),  # , camera_name='birdview'),
            env._target,
            done,
            skill if skill is None else SKILL2ID[skill],
            env.sim.data,
        )

        ns, _, _, _ = env.step(act)

        ts += 1
        if done:
            # fill final skill segment with goal reaching skill
            fill_skills(data, str(last_room_id)) if last_room_id != -1 else pad_skills(
                data
            )

            if len(data["actions"]) > args.min_traj_len:
                save_data(args, data, cnt)
                cnt += 1
            data = reset_data()
            env, controller = sample_env_and_controller(args, layout)
            s = reset_env(env, agent_centric=args.agent_centric)
            ts = 0
            last_room_id = None
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

    # add semantic skills
    traj_data.create_dataset("skills", data=data["skills"])

    f.close()


if __name__ == "__main__":
    main()
