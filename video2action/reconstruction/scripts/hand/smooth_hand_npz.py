#!/usr/bin/env python3
"""
Temporal de-jitter for HaWoR hand output (all_hand_meshes.npz), offline / zero-lag.

HaWoR predicts per-frame MANO; the trajectory jitters frame-to-frame. This smooths
it WITHOUT re-running HaWoR, in the camera frame (Task-4 consumes camera-frame data):
  * positions (trans / joints / vertices) -> forward-backward Savitzky-Golay
  * global rotation (rot, axis-angle)      -> sign-aligned windowed nlerp (quat)
  * finger pose (hand_pose)                -> light Savitzky-Golay
Only frames where <hand>_valid is True are smoothed; invalid spans are left as-is.

Output keys are IDENTICAL to the input npz (so downstream code is unchanged).

Usage:
  python smooth_hand_npz.py --input all_hand_meshes.npz --output all_hand_meshes_smoothed.npz \
      --pos-window 9 --pos-poly 2 --rot-window 3 --pose-window 5
"""
import argparse
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

HANDS = ["left", "right"]
POS_KEYS = ["trans", "joints", "vertices"]   # (N,3) / (N,J,3) / (N,V,3)


def _savgol(arr, window, poly):
    """Forward-backward Savitzky-Golay along axis 0. arr: (N, ...)."""
    n = arr.shape[0]
    if n < 3:
        return arr
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return arr
    p = min(poly, w - 1)
    flat = arr.reshape(n, -1)
    sm = savgol_filter(flat, window_length=w, polyorder=p, axis=0, mode="interp")
    return sm.reshape(arr.shape)


def _smooth_rotvec(rotvec, window):
    """rotvec: (N,3) axis-angle -> smoothed axis-angle via sign-aligned windowed
    nlerp on the quaternion manifold."""
    n = rotvec.shape[0]
    if n < 3 or window <= 0:
        return rotvec
    q = R.from_rotvec(rotvec).as_quat()          # (N,4) xyzw, unit
    for i in range(1, n):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    out = np.empty_like(q)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        m = q[lo:hi].mean(axis=0)
        nm = np.linalg.norm(m)
        out[i] = q[i] if nm == 0 else m / nm
    return R.from_quat(out).as_rotvec()


def _usable_mask(arr, valid):
    """A frame is usable only if flagged valid AND fully finite (HaWoR emits NaN for
    frames it couldn't reconstruct, sometimes even where <hand>_valid is True)."""
    n = arr.shape[0]
    finite = np.isfinite(arr.reshape(n, -1)).all(axis=1)
    return valid & finite


def _apply_on_valid(arr, valid, fn):
    """Apply fn over each contiguous usable (valid & finite) run; leave others as-is."""
    out = arr.copy()
    n = arr.shape[0]
    usable = _usable_mask(arr, valid)
    i = 0
    while i < n:
        if not usable[i]:
            i += 1
            continue
        j = i
        while j < n and usable[j]:
            j += 1
        if j - i >= 3:
            out[i:j] = fn(arr[i:j])
        i = j
    return out


def smooth_hand_npz(data, pos_window, pos_poly, rot_window, pose_window):
    out = {k: np.array(v) for k, v in data.items()}
    for hand in HANDS:
        vkey = f"{hand}_valid"
        valid = np.array(data[vkey]) if vkey in data else np.ones(
            data[f"{hand}_trans"].shape[0], dtype=bool)
        for pk in POS_KEYS:
            key = f"{hand}_{pk}"
            if key in out:
                out[key] = _apply_on_valid(out[key], valid, lambda a: _savgol(a, pos_window, pos_poly))
        rkey = f"{hand}_rot"
        if rkey in out:
            out[rkey] = _apply_on_valid(out[rkey], valid, lambda a: _smooth_rotvec(a, rot_window))
        pkey = f"{hand}_hand_pose"
        if pkey in out and pose_window > 0:
            out[pkey] = _apply_on_valid(out[pkey], valid, lambda a: _savgol(a, pose_window, 2))
    return out


def jitter_metric(data):
    """Mean 2nd-difference norm of right-hand joints (proxy for visible jitter)."""
    j = np.array(data["right_joints"])
    if j.shape[0] < 3:
        return 0.0
    d = np.linalg.norm(np.diff(j, 2, axis=0), axis=-1)
    d = d[np.isfinite(d).all(axis=-1)] if d.ndim > 1 else d[np.isfinite(d)]
    return float(d.mean()) if d.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pos-window", type=int, default=9)
    ap.add_argument("--pos-poly", type=int, default=2)
    ap.add_argument("--rot-window", type=int, default=3)
    ap.add_argument("--pose-window", type=int, default=5)
    args = ap.parse_args()

    data = dict(np.load(args.input))
    before = jitter_metric(data)
    out = smooth_hand_npz(data, args.pos_window, args.pos_poly, args.rot_window, args.pose_window)
    after = jitter_metric(out)
    np.savez(args.output, **out)
    print(f"jitter(right_joints 2nd-diff): {before:.5f} -> {after:.5f}  ({before/max(after,1e-9):.1f}x)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
