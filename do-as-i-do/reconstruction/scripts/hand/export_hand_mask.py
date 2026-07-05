#!/usr/bin/env python3
"""
Export the hand 2D mask (题目 3.2 子项1) as a video overlay + a compact .npy.

Mask definition (stated for grading): the *full hand-shape* mask = the silhouette
of the reconstructed MANO mesh (left ∪ right) rasterized into the image plane by
HaWoR (hawor_video.py writes tracks_S_E/model_masks.npy, shape (N,H,W) bool). This
is a completed/estimated full-hand mask, NOT a visible-region-only segmentation.

Produces:
  * hand_mask_overlay.mp4  — green mask tinted over the source frames + contour
  * hand_mask.npy          — (N,H,W) uint8 {0,1} copy of the mask (portable)

Usage:
  python export_hand_mask.py --video input.mp4 --model-masks tracks_0_265/model_masks.npy \
      --out-video hand_mask_overlay.mp4 --out-npy hand_mask.npy
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import cv2

TINT = np.array([0, 255, 0], dtype=np.float32)  # green (RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model-masks", required=True, help="HaWoR tracks_*/model_masks.npy")
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--out-npy", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    masks = np.load(args.model_masks)          # (N,H,W) bool/float
    masks = (np.asarray(masks) > 0).astype(np.uint8)
    N = masks.shape[0]

    if args.out_npy:
        np.save(args.out_npy, masks)

    cap = cv2.VideoCapture(args.video)
    tmp = tempfile.mkdtemp()
    written = 0
    for t in range(N):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        m = masks[t]
        if m.shape != rgb.shape[:2]:
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        a = (m.astype(np.float32) * args.alpha)[..., None]
        comp = rgb * (1 - a) + TINT[None, None, :] * a
        # draw contour for a crisp boundary
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comp = comp.astype(np.uint8)
        cv2.drawContours(comp, cnts, -1, (0, 200, 0), 2)
        cv2.imwrite(os.path.join(tmp, f"{t:04d}.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        written += 1
    cap.release()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_video)), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps),
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out_video],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"hand mask: {written} frames -> {args.out_video}"
          + (f" ; {args.out_npy}" if args.out_npy else ""))


if __name__ == "__main__":
    main()
