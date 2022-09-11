import pickle
import gzip

import numpy as np
import d4rl
import gym


target_path = "data/kitchen_states_no_goal.pkl.gz"
env = gym.make("kitchen-mixed-v0")
dataset = env.get_dataset()
L = len(dataset["actions"])

dataset_ep = []
obs = []
acs = []
dones = []
for i in range(L):
    done = dataset["terminals"][i] or dataset["timeouts"][i]
    obs.append(dataset["observations"][i][:30])  # Remove goal [30:60] from ob [0:60]
    acs.append(dataset["actions"][i])
    dones.append(done)
    if done:
        dones[-1] = True
        dataset_ep.append(
            dict(obs=np.array(obs), actions=np.array(acs), dones=np.array(dones))
        )
        print(f"{i}: saving episode of length {len(dones)}")
        obs = []
        acs = []
        dones = []

print(f"Storing dataset to file {target_path}...")
pickle.dump(dataset_ep, gzip.open(target_path, "wb"))

print(f"Loading dataset from file {target_path}...")
dataset_loaded = pickle.load(gzip.open(target_path, "rb"))

print(f"{len(dataset_loaded)} trajectories loaded from file {target_path}")
