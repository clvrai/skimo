import argparse
import d4rl
import gym
import numpy as np
import h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="maze2d-hardexp-v2")
    args = parser.parse_args()

    env = gym.make(args.env_name)
    render_env = gym.make(args.env_name[:-3] + "-agent_centric" + args.env_name[-3:])

    dataset = env.get_dataset()
    if "infos/qpos" not in dataset:
        raise ValueError("Only MuJoCo-based environments can be visualized")
    qpos = dataset["infos/qpos"]
    qvel = dataset["infos/qvel"]
    rewards = dataset["rewards"]
    actions = dataset["actions"]

    render_env.reset()
    render_env.set_state(qpos[0], qvel[0])
    # [render_env.render() for _ in range(100)]       # bring camera in position
    imgs = np.random.rand(1000000, 32, 32, 3)
    # for t in range(qpos.shape[0]):
    #     env.set_state(qpos[t], qvel[t])
    #     env.render()
    dataset["images"] = imgs
    dataset_new = h5py.File("joined_data.h5", "w")
    for k in dataset:
        dataset_new.create_dataset(k, data=dataset[k], compression="gzip")
    print("Done!")
