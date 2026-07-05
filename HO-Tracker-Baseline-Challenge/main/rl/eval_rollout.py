import argparse
import os
import re
import glob
import subprocess
from termcolor import cprint


def extract_epoch(path):
    match = re.search(r"ep_+(\d+)_+", path)
    if match:
        return int(match.group(1))
    return -1


def extract_reward(path):
    match = re.search(r"rew_+([-+]?\d*\.?\d+)_+", path)
    if match:
        return float(match.group(1))
    return -1


def run_exp(
    tag,
    index,
    gpu_id,
    side,
    dexhand,
    smooth,
    extra_args,
    checkpoint=None,
):
    if side == "rh":
        side_tag = "RH"
    elif side == "lh":
        side_tag = "LH"
    elif side == "bih":
        side_tag = "BiH"
    else:
        raise ValueError
    print(
        f"CUDA_VISIBLE_DEVICES={gpu_id} python main/rl/train.py task=ResDexHand dexhand={dexhand} side={side_tag} headless=true num_envs=256 test=true rolloutStateInit=false randomStateInit=false dataIndices=[{index}] checkpoint={checkpoint} actionsMovingAverage={smooth} experiment={tag} save_rollouts=true num_rollouts_to_save=512 num_rollouts_to_run=8192 save_successful_rollouts_only=false {' '.join(extra_args)}\n"
    )
    command = f"bash -c 'PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES={gpu_id} python main/rl/train.py task=ResDexHand dexhand={dexhand} side={side_tag} headless=true num_envs=256 test=true rolloutStateInit=false randomStateInit=false dataIndices=[{index}] checkpoint={checkpoint} actionsMovingAverage={smooth} experiment={tag} save_rollouts=true num_rollouts_to_save=512 num_rollouts_to_run=8192 save_successful_rollouts_only=false {' '.join(extra_args)}'"

    subprocess.run(command, shell=True, check=True, text=True)


def batch_rollout():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="baseline")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dexhand", type=str, default="inspire")
    parser.add_argument("--smooth", type=float, default=0.4)
    # user can also specify other arguments.
    parser.add_argument(
        "--extra",
        nargs="+",
        type=str,
        default=[],
        help="extra custom arguments (any length, including spaces)",
    )

    args = parser.parse_args()

    source_path = "runs"

    val_list = glob.glob("./data/HO-Tracker/data/test_sample/h*o*/*")
    val_list = [os.path.basename(val) for val in val_list]
    os.makedirs("dumps", exist_ok=True)

    todo_list = sorted(os.listdir(source_path))

    todo_list = [
        todo
        for todo in todo_list
        if args.tag in todo and args.dexhand in todo and os.path.isdir(f"{source_path}/{todo}")
    ]

    for i, todo in enumerate(todo_list):

        ckpts = glob.glob(f"{os.path.join(source_path, todo)}/nn/*.pth")
        ckpts = sorted(ckpts, key=os.path.getctime)
        if len(ckpts) == 0:
            continue

        file_infos = todo.split(f"{args.tag}_")[-1].split("_")

        seq_info, dexhand, side = file_infos[0], file_infos[1], file_infos[2]

        if seq_info not in val_list:
            continue

        # * Use the latest checkpoint
        # ckpt = ckpts[-1]
        # * Or, you can choose the checkpoint with the highest reward or the max epoch
        # reward = max([extract_reward(x) for x in ckpts])
        max_epoch = max([extract_epoch(x) for x in ckpts])
        ckpt = [x for x in ckpts if extract_epoch(x) == max_epoch][0]

        tag = args.tag

        run_exp(
            tag,
            seq_info,
            args.gpu_id,
            side,
            dexhand,
            args.smooth,
            args.extra,
            os.path.abspath(ckpt),
        )


if __name__ == "__main__":
    batch_rollout()