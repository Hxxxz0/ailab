#!/usr/bin/env python3
"""
Quality-control for reconstructed hands (题目 3.2 子项3 轨迹质量 & 子项4 运动学约束).

Reads the smoothed all_hand_meshes npz and reports, per hand:
  * valid frame count / ratio
  * bone-length consistency across frames (MANO bones should be near-constant;
    high variation => 骨长突变) — reported as coefficient of variation
  * trajectory jitter (mean 2nd-difference of joints & wrist) before smoothing vs after
  * anomaly frames (robust MAD outliers on joint 2nd-difference) — potential 跳变/翻折
  * chirality sign (left vs right must be OPPOSITE; equal => 左右手混淆)

Also writes a keyframe grid (sampled from an overlay video if given, else source).

Usage:
  python hand_qc.py --smoothed s.npz --raw r.npz --out qc_report.json \
      --grid keyframes_grid.png --overlay hand_mesh_overlay.mp4
"""
import argparse
import json
import numpy as np

# OpenPose-21 hand skeleton edges (HaWoR joints use mano_to_openpose ordering)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
MCP = {"index": 5, "pinky": 17, "thumb": 1}


def _second_diff_norm(x):
    """Per-frame 2nd-difference magnitude, dropping any non-finite entries."""
    if x.shape[0] < 3:
        return np.zeros(0)
    d = np.linalg.norm(np.diff(x, 2, axis=0), axis=-1)
    if d.ndim > 1:
        d = d.mean(axis=-1)
    return d[np.isfinite(d)]


def _bone_cov(joints):
    """Coefficient of variation of each bone length over valid frames, averaged."""
    lens = np.stack([np.linalg.norm(joints[:, a] - joints[:, b], axis=-1) for a, b in EDGES], axis=1)  # (N,E)
    mean = lens.mean(axis=0) + 1e-9
    cov = (lens.std(axis=0) / mean)
    return float(cov.mean()), float(cov.max())


def _chirality_sign(joints):
    """Median scalar triple product sign over frames (opposite for L vs R hand)."""
    w = joints[:, 0]
    v_i = joints[:, MCP["index"]] - w
    v_p = joints[:, MCP["pinky"]] - w
    v_t = joints[:, MCP["thumb"]] - w
    triple = np.einsum("ni,ni->n", np.cross(v_i, v_p), v_t)
    return float(np.sign(np.median(triple)))


def qc_hand(data, hand, raw=None):
    vkey, jkey, tkey = f"{hand}_valid", f"{hand}_joints", f"{hand}_trans"
    n = data[jkey].shape[0]
    valid = np.array(data[vkey]) if vkey in data else np.ones(n, dtype=bool)
    # usable = flagged valid AND finite (HaWoR emits NaN for unreconstructed frames)
    jarr = np.array(data[jkey])
    finite = np.isfinite(jarr.reshape(n, -1)).all(axis=1)
    usable = valid & finite
    rep = {"n_frames": int(n), "valid_frames": int(usable.sum()),
           "valid_ratio": float(usable.mean()),
           "flagged_valid_but_nan": int((valid & ~finite).sum())}
    if usable.sum() < 3:
        rep["note"] = "too few usable frames for kinematic stats"
        return rep
    j = jarr[usable]
    t = np.array(data[tkey])[usable]
    cov_mean, cov_max = _bone_cov(j)
    rep["bone_len_cov_mean"] = cov_mean
    rep["bone_len_cov_max"] = cov_max
    rep["chirality_sign"] = _chirality_sign(j)
    jd = _second_diff_norm(jarr)
    rep["joint_jitter_smoothed"] = float(jd.mean()) if jd.size else 0.0
    wd = _second_diff_norm(np.array(data[tkey]))
    rep["wrist_jitter_smoothed"] = float(wd.mean()) if wd.size else 0.0
    if raw is not None and jkey in raw:
        rjd = _second_diff_norm(np.array(raw[jkey]))
        rep["joint_jitter_raw"] = float(rjd.mean()) if rjd.size else 0.0
        rwd = _second_diff_norm(np.array(raw[tkey]))
        rep["wrist_jitter_raw"] = float(rwd.mean()) if rwd.size else 0.0
        denom = max(rep["joint_jitter_smoothed"], 1e-9)
        rep["jitter_reduction_x"] = round(rep["joint_jitter_raw"] / denom, 2)
    # anomaly frames: MAD outliers on raw joint 2nd-difference
    src = np.array(raw[jkey]) if raw is not None and jkey in raw else jarr
    d = _second_diff_norm(src)
    if d.size:
        med = np.median(d); mad = np.median(np.abs(d - med)) + 1e-9
        anomalies = np.where(d > med + 6 * 1.4826 * mad)[0] + 1  # +1: 2nd-diff index -> frame
        rep["anomaly_frames"] = anomalies.tolist()
        rep["n_anomaly_frames"] = int(anomalies.size)
    return rep


def make_grid(video, out_png, n=6):
    import cv2
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
    tiles = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            tiles.append(cv2.resize(f, (426, 240)))
    cap.release()
    if not tiles:
        return False
    cols = 3
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(out_png, np.vstack(rows))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoothed", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--overlay", default=None, help="video to sample keyframes from")
    ap.add_argument("--source", default=None, help="fallback source video for grid")
    ap.add_argument("--mask-definition", default=(
        "full hand-shape mask = rasterized silhouette of the reconstructed MANO mesh "
        "(left ∪ right), from HaWoR model_masks.npy; estimated full shape, not visible-only"))
    args = ap.parse_args()

    data = dict(np.load(args.smoothed))
    raw = dict(np.load(args.raw)) if args.raw else None
    report = {"mask_definition": args.mask_definition, "hands": {}}
    for hand in ["right", "left"]:
        if f"{hand}_joints" in data:
            report["hands"][hand] = qc_hand(data, hand, raw)
    # left/right confusion check
    rs = report["hands"].get("right", {}).get("chirality_sign")
    ls = report["hands"].get("left", {}).get("chirality_sign")
    if rs is not None and ls is not None and report["hands"]["left"]["valid_ratio"] > 0.1:
        report["chirality_consistent"] = bool(rs != ls)   # opposite signs expected
    else:
        report["chirality_consistent"] = None

    if args.grid:
        vid = args.overlay or args.source
        if vid and make_grid(vid, args.grid):
            report["keyframe_grid"] = args.grid

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"QC -> {args.out}")
    for h, r in report["hands"].items():
        cov = r.get("bone_len_cov_mean")
        cov_s = f"{cov:.4f}" if cov is not None else "-"
        print(f"  {h}: valid {r['valid_frames']}/{r['n_frames']}, "
              f"bone_cov={cov_s}, jitter_x={r.get('jitter_reduction_x','-')}, "
              f"anomalies={r.get('n_anomaly_frames','-')}")
    print(f"  chirality_consistent={report['chirality_consistent']}")


if __name__ == "__main__":
    main()
