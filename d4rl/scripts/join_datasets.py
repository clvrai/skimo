import argparse
import d4rl
import gym
import os
import numpy as np
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="maze2d-hardexp-v2")
parser.add_argument("--data_dir", type=str, default=".")
args = parser.parse_args()

env = gym.make(args.env_name)

fused_data = None

for root, dirs, files in os.walk(args.data_dir):
    for file in files:
        if file.endswith(".h5"):
            dataset = env.get_dataset(os.path.join(root, file))
            if fused_data is None:
                fused_data = dataset
            else:
                for key in fused_data:
                    fused_data[key] = np.concatenate((fused_data[key], dataset[key]))

dataset = h5py.File(os.path.join(root, "joined_data.h5"), "w")
for k in fused_data:
    dataset.create_dataset(k, data=fused_data[k], compression="gzip")
print("Done!")
