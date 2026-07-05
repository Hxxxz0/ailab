#!/usr/bin/env python3
"""
Read HO-Tracker camera intrinsics (cam_intr.pkl = 3x3 K matrix) into fx,fy,cx,cy,
and (optionally) the image size from the paired video.
"""
import pickle
import numpy as np


def load_intrinsics(cam_intr_pkl: str, video_path: str = None) -> dict:
    with open(cam_intr_pkl, "rb") as f:
        K = pickle.load(f)
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"expected 3x3 intrinsics, got {K.shape} from {cam_intr_pkl}")
    out = {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "K": K.tolist(),
    }
    if video_path:
        import cv2
        cap = cv2.VideoCapture(video_path)
        out["W"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out["H"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out["n_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-intr", required=True)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    print(json.dumps(load_intrinsics(args.cam_intr, args.video), indent=2))
