# Tracking Dexterous Hand–Object Manipulation from Human Demonstration

This repository is used for the **Video2Motion2Action** challenge assignment. It provides baseline implementations, data preparation scripts, and evaluation utilities. Participants are expected to set up the environment, run the provided Inspire sample, add Sharpa support, complete the required training and evaluation workflow, and improve the final score.

---

## 🛠️ Installation
<a id="Installation"></a>

<details>
<summary>Steps:</summary>

1. Clone the repository and initialize submodules:
    ```bash
    # Option A: clone from GitHub and initialize submodules
    git clone https://github.com/kelvin34501/HO-Tracker-Baseline-Challenge.git
    cd HO-Tracker-Baseline-Challenge
    git submodule init && git submodule update

    # Option B: extract the provided code tarball
    # No submodule initialization is required for the provided tarball.
    ```
2. Create a virtual environment named `maniptrans` with Python 3.8. Note that IsaacGym only supports Python versions up to 3.8.
    ```bash
    conda create -y -n maniptrans python=3.8
    conda activate maniptrans
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
3. Download IsaacGym Preview 4 from the [official website](https://developer.nvidia.com/isaac-gym) and follow the installation instructions in the documentation. Test the installation by running an example script, such as `joint_monkey.py`, located in the `python/examples` directory.
4. Install additional dependencies.
    ```bash
    pip install git+https://github.com/ZhengyiLuo/smplx.git
    pip install git+https://github.com/KailinLi/bps_torch.git
    pip install fvcore~=0.1.5
    pip install --no-index --no-cache-dir pytorch3d==0.7.3 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
    pip install -r requirements.txt
    pip install -e . # include the current directory in the Python path. Or use: `export PYTHONPATH=.:$PYTHONPATH`
    pip install numpy==1.23.5 # downgrade numpy to 1.23.5 to avoid compatibility issues
    ```

</details>

---

## 📋 Prerequisites
<a id="Prerequisites"></a>


### Download data for `HO-Tracker` dataset
Download the `HO-Tracker` dataset from [Hugging Face](https://huggingface.co/datasets/kelvin34501/HO-Tracker-Challenge) and extract it to `data/HO-Tracker` and `data/human_demo`.

### MISC
Refer to the [ManipTrans](https://github.com/ManipTrans/ManipTrans?tab=readme-ov-file#misc) README for additional prerequisites and environment notes.


## ▶️ Training
<a id="usage"></a>

The commands below provide the starter Inspire workflow for the sample sequence `0f900@0`. They are baseline examples for getting started; adapt workflow as needed for Sharpa and for the full challenge training requirements.

```bash
# To train the ManipTrans baseline:
python main/dataset/mano2dexhand.py --data_idx 0f900@0 --side left --dexhand inspire --headless --iter 7000

python main/rl/train.py task=ResDexHand dexhand=inspire side=LH headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[0f900@0] actionsMovingAverage=0.4 experiment=baseline
```

Train your model(s) on the `data/HO-Tracker/data/test_sample` set.

For the full challenge workflow, you may need to write your own parallel processing scripts to manage parallel jobs, including training runs and rollout saving across multiple sequences or experiment settings.

> **Note:** Evaluation compliance. Save checkpoints under `runs/` following the naming pattern: `runs/{your exp tag}_{seq id (do not modify)}_{dexhand (i.e. inspire)}_{hand side (e.g. rh, lh, or bih)}__{timestamp}/nn/last_{your exp tag}_ep_{#epoch}_xxxx.pth`.

## ▶️ Evaluation
<a id="eval"></a>

After training, evaluate your model with the official rollout and scoring path:

```bash
# To eval the ManipTrans baseline:
python main/rl/eval_rollout.py --tag baseline --dexhand inspire
# You can modify the arguments / rollout code according to your needs.
```

For scoring saved rollouts:
```bash
# To eval the scores of the saved rollouts:
python main/rl/eval_score.py
```

You will obtain summary metrics similar to:
```
================ Overall Results ================
Number of successful sequences: X
Average success rate: Single hand:  X, Bi-hand:  X
Average et (cm):  X
Average er (degree):  X
Average ej (cm):  X
Average eft (cm):  X
```

## ▶️ Sharpa workflow
<a id="sharpa"></a>

This repository includes a Sharpa Wave integration for the challenge tracking
task. The Sharpa URDF/STL assets are imported from
[`sharpa-robotics/sharpa-urdf-usd-xml`](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
at commit `6eea427eb24189519f32b9f21674cd534d3f973c` under Apache-2.0. See
`maniptrans_envs/assets/sharpa_wave/SOURCE.md`.

Before running IsaacGym scripts, set the extension cache and IsaacGym path:

```bash
cd HO-Tracker-Baseline-Challenge
conda activate maniptrans
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export PYTHONPATH=.:/mnt/workspace/Jensen/project/ailab/isaacgym/python:$PYTHONPATH
```

Verify the environment and Sharpa interface:

```bash
scripts/verify_maniptrans_env.sh
python scripts/check_sharpa_interface.py
```

For EGL mesh rendering in headless mode, this environment uses:

```bash
pip install --upgrade PyOpenGL==3.1.7 PyOpenGL-accelerate==3.1.7
```

Retarget one sequence and render a check video:

```bash
python main/dataset/mano2dexhand.py \
  --data_idx 0f900@0 --side left --dexhand sharpa --headless --iter 1000

PYOPENGL_PLATFORM=egl python scripts/render_retarget_mesh_video.py \
  --opt data/retargeting/HO-Tracker/mano2sharpa_lh/test_sample/h1o1/0f900@0/opt.pkl \
  --urdf maniptrans_envs/assets/sharpa_wave/sharpa_wave_left.urdf \
  --output videos/sharpa_0f900_left_mesh_local.mp4 \
  --root-mode local --source-order isaac
```

Batch retarget, train, rollout, and score:

```bash
ITER=1000 GPU_ID=0 scripts/retarget_sharpa_all.sh
EXP=sharpa_v1 GPU_ID=0 NUM_ENVS=4096 MAX_ITERATIONS=1000 scripts/train_sharpa_all.sh
EXP=sharpa_v1 GPU_ID=0 scripts/eval_sharpa_all.sh
```

Use `ITER=7000` only after the quick retarget video looks correct.
Optional retarget smoothing is available through `DOF_SMOOTH_WEIGHT` and
`DOF_ACC_WEIGHT`, but both default to `0.0` because over-smoothing can suppress
finger motion.

## Rules
<a id="rules"></a>

- Do not modify the official scoring logic.
- Do not hard-code evaluation outputs or sequence-specific answers.
- Do not submit only the unmodified Inspire baseline.
- Save checkpoints under `runs/` using the required naming pattern described above.
