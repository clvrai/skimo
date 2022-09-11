# Pre-trained Models

We provide pre-trained model checkpoints in the kitchen, maze, and calvin environments. Using these models you can skip the pre-training step and directly run SkiMo on downstream tasks.

To download model checkpoints, run:
```bash
# Maze
mkdir -p log/maze.skimo.pretrain.test.0/ckpt
cd log/maze.skimo.pretrain.test.0/ckpt
gdown 162cmAz9E9D3DyfUSihItI5gae9Z_DdoY
cd ../../..

# Kitchen
mkdir -p log/kitchen.skimo.pretrain.test.0/ckpt
cd log/kitchen.skimo.pretrain.test.0/ckpt
gdown 1LepFSrzgpmkaEReddM-zYEDJZn2dTrQ-
cd ../../..

# CALVIN
mkdir -p log/calvin.skimo.pretrain.test.0/ckpt
cd log/calvin.skimo.pretrain.test.0/ckpt
gdown 1pgVcOhGYc-Romehsk-_doB3kc4NgveEC
cd ../../..
```

Now, for downstream RL, you can simply run
```bash
# Maze
python run.py --config-name skimo_maze run_prefix=test gpu=0 wandb=true rolf.phase=rl rolf.pretrain_ckpt_path=log/maze.skimo.pretrain.test.0/ckpt/maze_ckpt_00000140000.pt

# Kitchen
python run.py --config-name skimo_kitchen run_prefix=test gpu=0 wandb=true rolf.phase=rl rolf.pretrain_ckpt_path=log/kitchen.skimo.pretrain.test.0/ckpt/kitchen_ckpt_00000085000.pt

# CALVIN
python run.py --config-name skimo_calvin run_prefix=test gpu=0 wandb=true rolf.phase=rl rolf.pretrain_ckpt_path=log/calvin.skimo.pretrain.test.0/ckpt/calvin_ckpt_00000085000.pt
```
