import argparse
import d4rl
import gym
import tqdm


START_POS = [27.0, 4.0]
TARGET_POS = [18.0, 8.0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="kitchen-mixed-v0")
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset()
    dataset = env.get_dataset()
    actions = dataset["actions"]
    # env.set_target(TARGET_POS)
    # env.reset_to_location(START_POS)
    for t in tqdm.tqdm(range(10000)):
        _, _, done, _ = env.step(actions[t])  # env.action_space.sample())
        env.render(mode="human")
        if done:
            env.reset()
            # env.set_target(TARGET_POS)
            # env.reset_to_location(START_POS)
