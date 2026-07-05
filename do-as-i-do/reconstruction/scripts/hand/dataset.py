#!/usr/bin/env python3
"""
Dataset enumeration for Task-2 batch hand reconstruction.

Walks the HO-Tracker human_demo dataset and yields one job per (sequence, camera)
video clip, resolving: source mp4, camera-calibration pkl, object type and the
anchor hand (from the sequence name).

Layout expected (see 题目 §五 / HO-Tracker-data):
  <data_root>/<sequence>/video/{camera_side_1,camera_side_2,camera_top}.mp4
  <data_root>/<sequence>/camera_calib/<camera>/{cam_intr.pkl,cam_extr.pkl}
  <data_root>/<sequence>/pose_3d.hdf5
"""
import os
import glob
from dataclasses import dataclass

CAMERAS = ["camera_side_1", "camera_side_2", "camera_top"]

# object keyword -> canonical object token (used for config.json / Task-4 mesh naming)
_OBJECT_KEYWORDS = [
    ("pipette", "pipette"),
    ("bread", "bread"),
    ("drink_ad", "drink_bottle_ad"),
    ("drink_yykx", "drink_bottle_yykx"),
    ("drink", "drink_bottle"),
]


@dataclass
class Clip:
    sequence: str            # full sequence dir name
    camera: str              # camera_side_1 / camera_side_2 / camera_top
    video: str               # absolute path to source mp4
    cam_intr: str            # absolute path to cam_intr.pkl
    cam_extr: str            # absolute path to cam_extr.pkl
    object_name: str         # canonical object token
    anchor_hand: str         # "left" | "right"

    @property
    def clip_id(self) -> str:
        return f"{self.sequence}/{self.camera}"


def infer_object(sequence: str) -> str:
    s = sequence.lower()
    for kw, token in _OBJECT_KEYWORDS:
        if kw in s:
            return token
    return "object"


def infer_anchor_hand(sequence: str) -> str:
    # "__left__" suffix marks a left-hand-anchored variant; default right.
    return "left" if "__left" in sequence.lower() else "right"


def enumerate_clips(data_root: str, cameras=None, sequences=None):
    """Yield Clip for every (sequence, camera) that has a source mp4 + intrinsics."""
    cameras = cameras or CAMERAS
    clips = []
    seq_dirs = sorted(
        d for d in glob.glob(os.path.join(data_root, "*"))
        if os.path.isdir(os.path.join(d, "video"))
    )
    for sd in seq_dirs:
        seq = os.path.basename(sd)
        if sequences and seq not in sequences:
            continue
        for cam in cameras:
            video = os.path.join(sd, "video", f"{cam}.mp4")
            cam_intr = os.path.join(sd, "camera_calib", cam, "cam_intr.pkl")
            cam_extr = os.path.join(sd, "camera_calib", cam, "cam_extr.pkl")
            if not (os.path.isfile(video) and os.path.isfile(cam_intr)):
                continue
            clips.append(Clip(
                sequence=seq, camera=cam, video=video,
                cam_intr=cam_intr, cam_extr=cam_extr,
                object_name=infer_object(seq), anchor_hand=infer_anchor_hand(seq),
            ))
    return clips


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    clips = enumerate_clips(args.data_root)
    print(f"{len(clips)} clips:")
    for c in clips:
        print(f"  {c.clip_id:60s} obj={c.object_name:18s} hand={c.anchor_hand}")
