#!/usr/bin/env python3
"""
Temporal smoothing for object-tracking layout.json (de-jitter, offline / zero-lag).

The tracker predicts each frame's pose independently (stochastic sampling + selection),
so the raw trajectory jitters frame-to-frame. This post-process smooths the trajectory
WITHOUT re-running tracking:

  * translation  -> forward-backward Savitzky-Golay (zero phase lag, preserves motion peaks)
  * quaternions  -> sign-aligned windowed nlerp average (moving average on the quat manifold)
  * scale        -> left untouched by default (usually --fix_scale_to_init_frame already)

It smooths EVERY translation-like and quat-like field present under each object's
`local_to_scene` (translation / translation_camera_frame / quat_wxyz / new_quat /
quat_xyzw / quat_wxyz_camera_frame), so whichever field the projector reads stays
consistent. Objects are grouped by mesh (or by absence thereof) and smoothed along
their own frame_idx ordering.

Usage:
  python smooth_layout.py --input  layout.json \
                          --output layout_smoothed.json \
                          --trans-window 7 --trans-poly 2 --quat-window 2

Then re-run run_project_mesh_combined.py pointing --json at the smoothed file.
"""
import argparse
import json
import copy
import re
import numpy as np
from scipy.signal import savgol_filter

TRANS_KEYS = ["translation", "translation_camera_frame"]
QUAT_KEYS = ["quat_wxyz", "new_quat", "quat_xyzw", "quat_wxyz_camera_frame"]


def smooth_translation(arr, window, poly):
    """arr: (N,3). Forward-backward Savitzky-Golay per axis. Falls back to raw / a
    plain moving average when there aren't enough frames for the requested window."""
    n = arr.shape[0]
    if n < 3:
        return arr
    w = min(window, n if n % 2 == 1 else n - 1)  # window must be odd and <= N
    if w < 3:
        return arr
    p = min(poly, w - 1)
    return savgol_filter(arr, window_length=w, polyorder=p, axis=0, mode="interp")


def smooth_quaternions(quats, window):
    """quats: (N,4) in whatever order (order preserved). Sign-align to remove the
    q/-q double cover, then replace each frame by the normalized mean of its
    [i-window, i+window] neighbourhood (nlerp moving average)."""
    n = quats.shape[0]
    q = quats.astype(np.float64).copy()
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q /= norms
    # hemisphere alignment: flip so each quat is on the same side as its predecessor
    for i in range(1, n):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    if window <= 0 or n < 3:
        return q
    out = np.empty_like(q)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        m = q[lo:hi].mean(axis=0)
        nm = np.linalg.norm(m)
        out[i] = q[i] if nm == 0 else m / nm
    return out


def group_key(obj):
    """Object identity ACROSS frames. mesh_obj is per-frame ("drink_bottle_frame7.obj"),
    so strip the frame token to collapse a single object's per-frame meshes into one
    trajectory. `index` here == frame_idx (per-frame), so it must NOT be used for grouping."""
    m = obj.get("mesh_obj") or obj.get("object_name")
    if m:
        m = re.sub(r"_frame\d+", "", m)          # drink_bottle_frame7.obj -> drink_bottle.obj
        m = re.sub(r"\d+(?=\.\w+$)", "", m)       # trailing digits before extension
        return m
    return "__all__"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="layout.json produced by track_object.py")
    ap.add_argument("--output", required=True, help="where to write the smoothed layout")
    ap.add_argument("--trans-window", type=int, default=7,
                    help="Savitzky-Golay window (odd) for translation. Larger = smoother.")
    ap.add_argument("--trans-poly", type=int, default=2, help="Savitzky-Golay polyorder.")
    ap.add_argument("--quat-window", type=int, default=2,
                    help="Half-window for quaternion moving average (0 disables rotation smoothing).")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    objects = data.get("objects", [])
    if not objects:
        raise SystemExit("No 'objects' in layout JSON.")

    # group objects that share the same mesh, order each group by frame_idx
    groups = {}
    for i, obj in enumerate(objects):
        groups.setdefault(group_key(obj), []).append(i)

    out = copy.deepcopy(data)
    n_trans_fields = n_quat_fields = 0
    for _, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: objects[i].get("frame_idx", i))
        lts_list = [objects[i].get("local_to_scene", {}) for i in idxs]
        if len(idxs) < 3:
            continue

        for key in TRANS_KEYS:
            if all(key in lts for lts in lts_list):
                arr = np.array([lts[key] for lts in lts_list], dtype=np.float64)
                sm = smooth_translation(arr, args.trans_window, args.trans_poly)
                for j, i in enumerate(idxs):
                    out["objects"][i]["local_to_scene"][key] = sm[j].tolist()
                n_trans_fields += 1

        for key in QUAT_KEYS:
            if all(key in lts for lts in lts_list):
                arr = np.array([lts[key] for lts in lts_list], dtype=np.float64)
                sm = smooth_quaternions(arr, args.quat_window)
                for j, i in enumerate(idxs):
                    out["objects"][i]["local_to_scene"][key] = sm[j].tolist()
                n_quat_fields += 1

    out["note"] = (out.get("note", "") +
                   f" | smoothed: trans savgol(w={args.trans_window},p={args.trans_poly}), "
                   f"quat window={args.quat_window}").strip(" |")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Smoothed {len(groups)} object group(s), "
          f"{n_trans_fields} translation field(s) + {n_quat_fields} quat field(s).")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
