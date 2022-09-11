import os
import pickle
import gzip

import h5py
import numpy as np
from tqdm import tqdm


ENV = "calvin"
OB = "states"

assert OB in ["states", "images"]

data_dir = f"data/{ENV}/D_D"
target_path = f"data/{ENV}_{OB}.pkl.gz"

filenames = []
for subdir in os.listdir(data_dir):
    for root, dirs, files in os.walk(f"{data_dir}/{subdir}"):
        for file in files:
            if file.endswith(".h5"):
                filenames.append(os.path.join(root, file))

print(f"{len(filenames)} files in {data_dir}")

dataset = []
for file in tqdm(filenames):
    with h5py.File(file, "r") as F:
        if OB == "states":
            ob = F["traj0"][OB][()].astype(np.float32)
        elif OB == "images":
            ob = F["traj0"][OB][()].astype(np.uint8)
        ac = F["traj0"]["actions"][()].astype(np.float32)
        done = 1 - F["traj0"]["pad_mask"][()].astype(np.float32)
        print(f"\t{file}: Load ob ({ob.shape})  ac ({ac.shape})")
        dataset.append(dict(obs=ob, actions=ac, dones=done))


print(f"Storing dataset to file {target_path}...")
pickle.dump(dataset, gzip.open(target_path, "wb"))

print(f"Loading dataset from file {target_path}...")
dataset_loaded = pickle.load(gzip.open(target_path, "rb"))

print(f"{len(dataset_loaded)} trajectories loaded from file {target_path}")
