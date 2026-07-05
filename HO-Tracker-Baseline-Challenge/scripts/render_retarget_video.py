#!/usr/bin/env python
import argparse
import os
import pickle
import subprocess
import tempfile

import isaacgym  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import maniptrans_envs.lib.envs.dexhands  # noqa: F401
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def _set_equal_axes(ax, xyz):
    center = xyz.reshape(-1, 3).mean(axis=0)
    span = np.ptp(xyz.reshape(-1, 3), axis=0).max()
    span = max(float(span), 0.15)
    half = span * 0.65
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def render(opt_path, output, dexhand, side, fps, stride, max_frames):
    hand = DexHandFactory.create_hand(dexhand, side)
    with open(opt_path, "rb") as f:
        data = pickle.load(f)
    joints = np.asarray(data["opt_joints_pos"], dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected opt_joints_pos shape [T, J, 3], got {joints.shape}")
    frame_ids = np.arange(0, len(joints), stride)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="retarget_frames_") as frame_dir:
        for frame_id in frame_ids:
            xyz = joints[frame_id]
            fig = plt.figure(figsize=(7, 7), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=18, c="#1f77b4")
            for a, b in hand.bone_links:
                if a < len(xyz) and b < len(xyz):
                    ax.plot(
                        [xyz[a, 0], xyz[b, 0]],
                        [xyz[a, 1], xyz[b, 1]],
                        [xyz[a, 2], xyz[b, 2]],
                        color="#333333",
                        linewidth=2,
                    )
            _set_equal_axes(ax, xyz)
            ax.set_title(f"{dexhand} {side} retarget frame {frame_id}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=22, azim=-65)
            fig.tight_layout()
            frame_path = os.path.join(frame_dir, f"{len(os.listdir(frame_dir)):06d}.png")
            fig.savefig(frame_path)
            plt.close(fig)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(frame_dir, "%06d.png"),
                "-pix_fmt",
                "yuv420p",
                output,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"Wrote {output} ({len(frame_ids)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True, help="Path to mano2dexhand opt.pkl")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--dexhand", default="sharpa")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    render(args.opt, args.output, args.dexhand, args.side, args.fps, args.stride, args.max_frames)


if __name__ == "__main__":
    main()
