# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Video2Motion2Action** challenge baseline for tracking dexterous hand–object manipulation from human demonstrations. The system retargets captured human hand motion (MANO) to dexterous robot hands and trains RL policies to reproduce those motions in IsaacGym simulation.

## Environment Setup

- **Python 3.8 required** (IsaacGym constraint — hard limit)
- Conda env name: `maniptrans`
- Key constraint: `numpy==1.23.5` must be pinned (downgraded after other installs)
- PyTorch: `1.13.1+cu117`

Installation order matters:
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# Install IsaacGym Preview 4 manually from NVIDIA
pip install git+https://github.com/ZhengyiLuo/smplx.git
pip install git+https://github.com/KailinLi/bps_torch.git
pip install fvcore~=0.1.5
pip install --no-index --no-cache-dir pytorch3d==0.7.3 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
pip install -r requirements.txt
pip install -e .
pip install numpy==1.23.5
```

Always set `PYTHONPATH=.:$PYTHONPATH` (or use `pip install -e .`).

## Data

- `data/HO-Tracker/` — challenge dataset (from Hugging Face: `kelvin34501/HO-Tracker-Challenge`)
- `data/human_demo/` — human demonstration sequences
- `data/HO-Tracker/data/test_sample/` — the sequences to train on for the challenge

## Commands

### Step 1: Preprocess / retarget motion data
```bash
python main/dataset/mano2dexhand.py --data_idx 0f900@0 --side left --dexhand inspire --headless --iter 7000
```

### Step 2: Train RL policy
```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=LH headless=true num_envs=4096 \
  learning_rate=2e-4 test=false randomStateInit=true dataIndices=[0f900@0] \
  actionsMovingAverage=0.4 experiment=baseline
```

### Step 3: Evaluate — save rollouts
```bash
python main/rl/eval_rollout.py --tag baseline --dexhand inspire
```

### Step 4: Score saved rollouts
```bash
python main/rl/eval_score.py
```

## Checkpoint Naming Convention (required for evaluation compliance)

```
runs/{exp_tag}_{seq_id}_{dexhand}_{side}__{timestamp}/nn/last_{exp_tag}_ep_{epoch}_xxxx.pth
```

Example: `runs/baseline_0f900@0_inspire_lh__20240101_120000/nn/last_baseline_ep_500_rew_1.234.pth`

Do not modify the seq_id portion of the path.

## Architecture

### Configuration (`main/cfg/`)
Hydra-based config. Top-level entry is `main/cfg/config.yaml`. Task configs live in `task/` and RL hyperparams in `rl_train/`. Key overridable params at CLI: `task`, `dexhand`, `side` (LH/RH/BiH), `num_envs`, `dataIndices`, `experiment`, `checkpoint`.

### Dataset pipeline (`main/dataset/`)
- `ManipDataFactory` — registry pattern; datasets self-register via `@register_manipdata("hotracker_rh")` decorator
- `ManipData` (base.py) — abstract `Dataset` base class; subclasses implement `__getitem__`
- `HOTracker` (ho_tracker.py) — main challenge dataset class (currently unregistered — register it to use)
- `mano2dexhand.py` — standalone preprocessing: optimizes dexhand joint angles to match MANO poses using IK, saves result to disk before RL training

### Dex Hand abstraction (`maniptrans_envs/lib/envs/dexhands/`)
- `DexHandFactory` — registry pattern; hands self-register via decorator in each hand file
- Currently supported: `inspire`, `shadow`, `allegro`, `xhand`, `artimano`
- Adding a new hand (e.g., Sharpa): create a new file, define left/right subclasses, register with `@register_dexhand`

### IsaacGym Environments (`maniptrans_envs/lib/envs/tasks/`)
- `DexHandManipRHEnv` / `DexHandManipLHEnv` — single-hand RL environment (dexhandmanip_sh.py)
- `DexHandManipBiHEnv` — bimanual environment (dexhandmanip_bih.py)
- All extend `VecTask` (core/vec_task.py), the GPU-vectorized environment base

### RL (`lib/rl/`)
- `PPOAgent` — extends rl_games' `ContinuousA2CBase`; used via the rl_games runner
- Network builders in `lib/rl/network_builder*.py` (residual architectures for SH and BiH)
- `lib/nn/` — MLP and LipsNet (Lipschitz-constrained network) building blocks

## Challenge Task

The baseline only implements the **Inspire** dexterous hand. The challenge requires:
1. Adding **Sharpa** hand support (new dexhand class + registration)
2. Training on the full `test_sample` sequences (not just the single `0f900@0` example)
3. Improving evaluation metrics (translation error `et`, rotation error `er`, joint error `ej`, fingertip error `eft`)

**Do not modify** `main/rl/eval_score.py` (official scoring logic).
