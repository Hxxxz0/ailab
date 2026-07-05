#!/usr/bin/env python
"""Render the source MANO hand (skeleton) from a HO-Tracker *_hand.pkl.

Companion to scripts/render_retarget_mesh_video.py: it draws the *input* MANO
motion so it can be compared against the retargeted dexhand meshes in
videos/sharpa_mesh_checks/.  The MANO mesh model files are not shipped with the
dataset, so we render the 21-joint hand skeleton straight from the precomputed
`mano_joints` keypoints (plus `wrist_pos` as the root), which needs no model.

root-mode matches the retarget videos:
  local    -> express joints in the wrist frame (wrist fixed at origin); only
              finger articulation moves.  Use this to compare against *_local.mp4.
  world    -> keep global coordinates (hand also translates/rotates in space).
"""
import argparse
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

# Ordered joints: wrist first, then each finger proximal->intermediate->distal->tip.
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
SEGMENTS = ["proximal", "intermediate", "distal", "tip"]
FINGER_COLORS = {
    "thumb": "#e6194B",
    "index": "#3cb44b",
    "middle": "#4363d8",
    "ring": "#f58231",
    "pinky": "#911eb4",
}


def _joint_order():
    names = ["wrist"]
    for finger in FINGERS:
        for seg in SEGMENTS:
            names.append(f"{finger}_{seg}")
    return names


def _bone_links(names):
    idx = {n: i for i, n in enumerate(names)}
    links = []
    for finger in FINGERS:
        links.append((idx["wrist"], idx[f"{finger}_proximal"]))
        for a, b in zip(SEGMENTS[:-1], SEGMENTS[1:]):
            links.append((idx[f"{finger}_{a}"], idx[f"{finger}_{b}"]))
    return links


def _load_joints(hand_path, root_mode):
    with open(hand_path, "rb") as f:
        data = pickle.load(f)
    mano_joints = data["mano_joints"]
    wrist_pos = np.asarray(data["wrist_pos"], dtype=np.float32)  # [T, 3]
    wrist_rot = np.asarray(data["wrist_rot"], dtype=np.float32)  # [T, 3] axis-angle
    T = wrist_pos.shape[0]

    names = _joint_order()
    joints = np.zeros((T, len(names), 3), dtype=np.float32)
    joints[:, 0] = wrist_pos
    for j, name in enumerate(names[1:], start=1):
        joints[:, j] = np.asarray(mano_joints[name], dtype=np.float32)

    if root_mode == "local":
        rot = Rotation.from_rotvec(wrist_rot).as_matrix()  # [T, 3, 3]
        rel = joints - wrist_pos[:, None, :]
        joints = np.einsum("tij,tkj->tki", rot.transpose(0, 2, 1), rel)
    elif root_mode != "world":
        raise ValueError(f"Unknown root_mode: {root_mode}")
    return joints, names, str(data.get("description", ""))


def _set_equal_axes(ax, xyz):
    center = xyz.reshape(-1, 3).mean(axis=0)
    span = max(float(np.ptp(xyz.reshape(-1, 3), axis=0).max()), 0.16)
    half = span * 0.6
    for setlim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setlim(c - half, c + half)


def render_mano_video(hand_path, output, fps, stride, max_frames, root_mode, width, height):
    joints, names, description = _load_joints(hand_path, root_mode)
    links = _bone_links(names)

    frame_ids = np.arange(0, len(joints), stride)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    # Fix the camera box from the whole (subsampled) clip so playback doesn't drift.
    box = joints[frame_ids]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mano_frames_") as frame_dir:
        for out_idx, frame_id in enumerate(frame_ids):
            xyz = joints[frame_id]
            fig = plt.figure(figsize=(width / 120, height / 120), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=16, c="#222222")
            for a, b in links:
                finger = names[b].split("_")[0]
                ax.plot(
                    [xyz[a, 0], xyz[b, 0]],
                    [xyz[a, 1], xyz[b, 1]],
                    [xyz[a, 2], xyz[b, 2]],
                    color=FINGER_COLORS[finger],
                    linewidth=2.5,
                )
            _set_equal_axes(ax, box)
            ax.set_title(f"MANO {description} frame {frame_id} ({root_mode})", fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            fig.savefig(Path(frame_dir) / f"{out_idx:06d}.png")
            plt.close(fig)

        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", str(Path(frame_dir) / "%06d.png"),
                "-pix_fmt", "yuv420p", str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"Wrote {output} ({len(frame_ids)} frames, root_mode={root_mode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True, help="Path to a HO-Tracker *_hand.pkl")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--root-mode", choices=["local", "world"], default="local")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    render_mano_video(
        args.hand, args.output, args.fps, args.stride,
        args.max_frames, args.root_mode, args.width, args.height,
    )


if __name__ == "__main__":
    main()
