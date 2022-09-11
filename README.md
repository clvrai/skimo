# Skill-based Model-based Reinforcement learning (SkiMo)

[[Project website](https://clvrai.com/skimo)] [[Paper](https://openreview.net/forum?id=iVxy2eO601U)] [[arXiv](https://arxiv.org/abs/2207.07560)]

This project is a PyTorch implementation of [Skill-based Model-based Reinforcement Learning](https://clvrai.com/skimo), published in CoRL 2022.


## Files and Directories
* `run.py`: launches an appropriate trainer based on algorithm
* `skill_trainer.py`: trainer for skill-based approaches
* `skimo_agent.py`: model and training code for SkiMo
* `skimo_rollout.py`: rollout with SkiMo agent
* `spirl_tdmpc_agent.py`: model and training code for SPiRL+TD-MPC
* `spirl_tdmpc_rollout.py`: rollout with SPiRL+TD-MPC
* `spirl_dreamer_agent.py`: model and training code for SPiRL+Dreamer
* `spirl_dreamer_rollout.py`: rollout with SPiRL+Dreamer
* `spirl_trainer.py`: trainer for SPiRL
* `spirl_agent.py`: model for SPiRL
* `config/`: default hyperparameters
* `calvin/`: CALVIN environments
* `d4rl/`: [D4RL](https://github.com/kpertsch/d4rl) environments forked by Karl Pertsch. The only change from us is in the [installation](d4rl/setup.py#L15) command
* `envs/`: environment wrappers
* `spirl/`: [SPiRL code](https://github.com/clvrai/spirl)
* `data/`: offline data directory
* `rolf/`: implementation of RL algorithms from [robot-learning](https://github.com/youngwoon/robot-learning) by Youngwoon Lee
* `log/`: training log, evaluation results, checkpoints


## Prerequisites
* Ubuntu 20.04
* Python 3.9
* MuJoCo 2.1


## Installation

1. Clone this repository.
```bash
git clone --recursive git@github.com:clvrai/skimo.git
cd skimo
```

2. Create a virtual environment
```bash
conda create -n skimo_venv python=3.9
conda activate skimo_venv
```

3. Install MuJoCo 2.1
* Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
* Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

4. Install packages
```bash
sh install.sh
```


## Download Offline Datasets

```bash
# Navigate to the data directory
mkdir data && cd data

# Maze
gdown 1GWo8Vr8Xqj7CfJs7TaDsUA6ELno4grKJ

# Kitchen (and mis-aligned kitchen)
gdown 1Fym9prOt5Cu_I73F20cdd3lXZPhrvEsd

# CALVIN
gdown 1g4ONf_3cNQtrZAo2uFa_t5MOopSr2DNY

cd ..
```


## Usage
Commands for SkiMo and all baselines. Results will be logged to [WandB](https://wandb.ai/site). Before running the commands below, **please change the wandb entity** in [run.py#L36](run.py#L36) to match your account.

### Environment

Please replace `[ENV]` with one of `maze`, `kitchen`, `calvin`. For mis-aligned kitchen, append `env.task=misaligned` to the downstream RL command.
After pre-training, please set `[PRETRAINED_CKPT]` with the proper path to the checkpoint.

### SkiMo (Ours)
* Pre-training
```bash
python run.py --config-name skimo_[ENV] run_prefix=test gpu=0 wandb=true
```
You can also skip this step by downloading our pre-trained model checkpoints. See instructions in [pretrained_models.md](pretrained_models.md).

* Downstream RL
```bash
python run.py --config-name skimo_[ENV] run_prefix=test gpu=0 wandb=true rolf.phase=rl rolf.pretrain_ckpt_path=[PRETRAINED_CKPT]
```

### Dreamer
```bash
python run.py --config-name dreamer_config env=[ENV] run_prefix=test gpu=0 wandb=true
```

### TD-MPC
```bash
python run.py --config-name tdmpc_config env=[ENV] run_prefix=test gpu=0 wandb=true
```

### SPiRL
* Need to first pre-train or download the skill prior (see instructions [here](https://github.com/clvrai/spirl#example-commands)).
* Downstream RL
```bash
python run.py --config-name spirl_config env=[ENV] run_prefix=test gpu=0 wandb=true
```

### SPiRL+Dreamer
* Downstream RL
```bash
python run.py --config-name spirl_dreamer_[ENV] run_prefix=test gpu=0 wandb=true
```

### SPiRL+TD-MPC
* Downstream RL
```bash
python run.py --config-name spirl_tdmpc_[ENV] run_prefix=test gpu=0 wandb=true
```

### SkiMo+SAC
* Downstream RL
```bash
python run.py --config-name skimo_[ENV] run_prefix=sac gpu=0 wandb=true rolf.phase=rl rolf.use_cem=false rolf.n_skill=1 rolf.prior_reg_critic=true rolf.sac=true rolf.pretrain_ckpt_path=[PRETRAINED_CKPT]
```

### SkiMo w/o joint training
* Pre-training
```bash
python run.py --config-name skimo_[ENV] run_prefix=no_joint gpu=0 wandb=true rolf.joint_training=false
```

* Downstream RL
```
python run.py --config-name skimo_[ENV] run_prefix=no_joint gpu=0 wandb=true rolf.joint_training=false rolf.phase=rl rolf.pretrain_ckpt_path=[PRETRAINED_CKPT]
```


## Troubleshooting

### Failed building wheel for mpi4py
Solution: install `mpi4py` with conda instead, which requires a lower version of python.
```bash
conda install python==3.8
conda install mpi4py
```
Now you can re-run `sh install.sh`.

### MacOS mujoco-py compilation error
See [this](https://github.com/openai/mujoco-py#youre-on-macos-and-you-see-clang-error-unsupported-option--fopenmp). In my case, I needed to change `/usr/local/` to `/opt/homebrew/` in all paths.


## Citation

If you find our code useful for your research, please cite:
```
@inproceedings{shi2022skimo,
  title={Skill-based Model-based Reinforcement Learning},
  author={Lucy Xiaoyang Shi and Joseph J. Lim and Youngwoon Lee},
  booktitle={Conference on Robot Learning},
  year={2022}
}
```


## References
* This code is based on Youngwoon's robot-learning repo: https://github.com/youngwoon/robot-learning
* SPiRL: https://github.com/clvrai/spirl
* TD-MPC: https://github.com/nicklashansen/tdmpc
* Dreamer: https://github.com/danijar/dreamer
* D4RL: https://github.com/rail-berkeley/d4rl
* CALVIN: https://github.com/mees/calvin
