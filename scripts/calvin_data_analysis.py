import numpy as np
from pathlib import Path

only_target = True
target_tasks = [
    "open_drawer",
    "close_drawer",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "move_slider_left",
    "move_slider_right",
]
idx_task_list = []
ep_start_end_list = []

# load task_id in the calvin dataset
for dataset in ["training", "validation"]:
    root_dir = Path(f"calvin/dataset/task_D_D/{dataset}")
    lang_data = np.load(
        root_dir / "lang_annotations/auto_lang_ann.npy", allow_pickle=True
    ).reshape(-1)[0]

    for ep in np.load(root_dir / "ep_start_end_ids.npy"):
        ep_start_end_list.append(ep)

    task_start_end_ids = lang_data["info"]["indx"]
    task = lang_data["language"]["task"]
    all_tasks = set(task)

    for idx, t in zip(task_start_end_ids, task):
        idx_task_list.append({"ep_idx": idx, "task": t})

# sort tasks by episode indices
idx_task_list.sort(key=lambda x: x.get("ep_idx"))
ep_start_end_list.sort(key=lambda x: x[0])
# print(sum([idx_task_list[i]['ep_idx'][1] - idx_task_list[i]['ep_idx'][0] for i in range(len(idx_task_list))]) / len(idx_task_list))

for n, ep in enumerate(ep_start_end_list):
    start = ep[0]
    closest_idx = 0
    closest_ep = None
    min_distance = 1000
    for i, e in enumerate(idx_task_list):
        if e["ep_idx"][0] <= start and e["ep_idx"][1] >= start:
            closest_ep = e
            closest_idx = i
            break
        elif e["ep_idx"][0] > start:
            # find the nearest
            if e["ep_idx"][0] - start < min_distance:
                closest_idx = i
                closest_ep = e
                min_distance = e["ep_idx"][0] - start
    if closest_ep["task"] in target_tasks or not only_target:
        curr_task = idx_task_list[closest_idx]["task"]
        print(n, curr_task, end="")

        for i in range(3):
            closest_idx += 1
            next_task = idx_task_list[closest_idx]["task"]
            while next_task == curr_task:
                closest_idx += 1
                next_task = idx_task_list[closest_idx]["task"]
            print(f" -> {next_task}", end="")
            curr_task = next_task
        print()

# define a task by its own frequency and task transition
class Task:
    def __init__(self, name):
        self.name = name
        self.tasks = target_tasks if only_target else all_tasks
        self.next = {k: 0 for k in self.tasks}
        self.next_freq = {k: 0.0 for k in self.tasks}
        self.instance = 0
        self.frequency = 0

    def add_next(self, next_task):
        self.next[next_task] += 1

    def compute_frequency(self):
        for k in self.tasks:
            self.next_freq[k] = self.next[k] / self.instance
        self.frequency = self.instance / len(idx_task_list)


tasks = target_tasks if only_target else all_tasks
task_dict = {k: Task(k) for k in tasks}
i = 0

while i < len(idx_task_list) - 1:
    curr_task = idx_task_list[i]["task"]
    i += 1

    # skip if only concerned about the target tasks
    if only_target and curr_task not in target_tasks:
        continue

    task_dict.get(curr_task).instance += 1
    next_task = idx_task_list[i]["task"]

    # tasks could repeat in a sequence
    while next_task == curr_task and i < len(idx_task_list) - 1:
        i += 1
        next_task = idx_task_list[i]["task"]

    if not only_target or next_task in target_tasks:
        task_dict.get(curr_task).add_next(next_task)

# compute the probability of each transition
prob_list = []
for k, v in task_dict.items():
    v.compute_frequency()
    for n, p in v.next_freq.items():
        prob_list.append({"curr": k, "next": n, "prob": v.frequency * p})

prob_list.sort(key=lambda x: x.get("prob"), reverse=True)
for t in prob_list:
    print(f"{t['curr']} -> {t['next']}: {t['prob'] * 100:.3f}%")
