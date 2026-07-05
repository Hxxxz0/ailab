#!/usr/bin/env python3
"""
Headless MANO hand overlay for HaWoR output (replaces the aitviewer/OpenGL step that
needs a GL context). Renders the reconstructed left/right hand meshes from
all_hand_meshes.npz onto the source video with PyTorch3D (works headless).

The npz vertices (left_vertices/right_vertices) are already in the OpenCV camera frame
(x-right, y-down, z-fwd) — the same convention as run_project_mesh_combined's `verts_cam` —
so we just apply cam_to_pytorch3d and render with the given focal length.

Usage (in the sam3d env):
  python render_hands_overlay.py \
      --video  camera_side_1.mp4 \
      --npz    camera_side_1/all_hand_meshes.npz \
      --output camera_side_1/hand_overlay.mp4 \
      --focal 922
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import torch
import cv2
from pytorch3d.renderer import (
    PerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex,
)
from pytorch3d.structures import Meshes

RIGHT_COLOR = (0.35, 0.55, 0.95)   # RGB in [0,1] — blue-ish right hand
LEFT_COLOR = (0.95, 0.60, 0.30)    # orange-ish left hand


def cam_to_pytorch3d(v):
    out = torch.zeros_like(v)
    out[:, 0] = -v[:, 0]
    out[:, 1] = -v[:, 1]
    out[:, 2] = v[:, 2]
    return out


def make_renderer(fx, fy, cx, cy, width, height, device):
    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=((height, width),), in_ndc=False, device=device,
    )
    raster = RasterizationSettings(image_size=(height, width), blur_radius=1e-5,
                                   faces_per_pixel=8, bin_size=None, max_faces_per_bin=200000)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )


def build_mesh(verts_np, faces_np, color, device):
    v = torch.tensor(verts_np, dtype=torch.float32, device=device)
    v = cam_to_pytorch3d(v)
    f = torch.tensor(faces_np.astype(np.int64), dtype=torch.int64, device=device)
    col = torch.tensor(color, dtype=torch.float32, device=device).view(1, 1, 3).expand(1, v.shape[0], 3)
    return Meshes(verts=[v], faces=[f], textures=TexturesVertex(verts_features=col))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--focal", type=float, default=922.0)
    ap.add_argument("--fx", type=float, default=None, help="horizontal focal (px); overrides --focal")
    ap.add_argument("--fy", type=float, default=None, help="vertical focal (px); overrides --focal")
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.75, help="overlay opacity")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.npz)
    T = d["right_vertices"].shape[0]

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fx = args.fx if args.fx is not None else args.focal
    fy = args.fy if args.fy is not None else args.focal
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    renderer = make_renderer(fx, fy, cx, cy, W, H, device)
    print(f"video {W}x{H}, {T} frames, fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    tmp = tempfile.mkdtemp()
    n_written = 0
    for t in range(T):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        comp = frame.copy()
        for side, vkey, fkey, valkey, color in [
            ("right", "right_vertices", "right_faces", "right_valid", RIGHT_COLOR),
            ("left", "left_vertices", "left_faces", "left_valid", LEFT_COLOR),
        ]:
            valid = bool(d[valkey][t]) if valkey in d else True
            # HaWoR emits NaN for frames it couldn't reconstruct (even some flagged valid)
            if not valid or not np.isfinite(d[vkey][t]).all():
                continue
            mesh = build_mesh(d[vkey][t], d[fkey], color, device)
            with torch.no_grad():
                img = renderer(mesh)[0].cpu().numpy()
            rgb, a = img[..., :3], img[..., 3]
            a = np.clip(a, 0, 1)[..., None] * args.alpha
            comp = comp * (1 - a) + rgb * a
        out = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tmp, f"{t:04d}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        n_written += 1
        if t % 30 == 0:
            print(f"  rendered {t}/{T}")
    cap.release()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps),
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", args.output], check=True)
    print(f"Saved {n_written} frames -> {args.output}")


if __name__ == "__main__":
    main()
