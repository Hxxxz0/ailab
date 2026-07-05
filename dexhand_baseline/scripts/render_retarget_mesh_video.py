#!/usr/bin/env python
import argparse
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from urdfpy import URDF


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def _load_motion(opt_path):
    with open(opt_path, "rb") as f:
        data = pickle.load(f)
    dof_pos = np.asarray(data["opt_dof_pos"], dtype=np.float32)
    wrist_pos = np.asarray(data["opt_wrist_pos"], dtype=np.float32)
    wrist_rot = np.asarray(data["opt_wrist_rot"], dtype=np.float32)
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected opt_dof_pos shape [T, D], got {dof_pos.shape}")
    return dof_pos, wrist_pos, wrist_rot


def _load_isaac_dof_names(urdf_path):
    import isaacgym  # noqa: F401
    from isaacgym import gymapi

    urdf_path = Path(urdf_path)
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, gymapi.SimParams())
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = False
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, asset_options)
    names = [gym.get_asset_dof_name(asset, i) for i in range(gym.get_asset_dof_count(asset))]
    gym.destroy_sim(sim)
    return names


def _make_cfg(source_names, target_names, dof_values):
    values_by_name = {name: float(value) for name, value in zip(source_names, dof_values)}
    missing = [name for name in target_names if name not in values_by_name]
    if missing:
        raise ValueError(f"Source DOF order is missing joints required by URDF FK: {missing}")
    return {name: values_by_name[name] for name in target_names}


def _root_transform(wrist_pos, wrist_rot, root_mode):
    root = np.eye(4, dtype=np.float64)
    if root_mode == "world":
        root[:3, :3] = Rotation.from_rotvec(wrist_rot).as_matrix()
        root[:3, 3] = wrist_pos
    elif root_mode == "rotation":
        root[:3, :3] = Rotation.from_rotvec(wrist_rot).as_matrix()
    elif root_mode != "local":
        raise ValueError(f"Unknown root_mode: {root_mode}")
    return root


def _make_scene(width, height, center, radius):
    scene = pyrender.Scene(bg_color=[248, 248, 248, 255], ambient_light=[0.45, 0.45, 0.45])
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(28.0))
    eye = center + np.array([0.22, -0.42, 0.18]) * max(radius / 0.18, 1.0)
    camera_pose = _look_at(eye, center)
    scene.add(camera, pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.5), pose=camera_pose)
    fill_pose = _look_at(center + np.array([-0.25, 0.2, 0.25]), center)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=fill_pose)
    renderer = pyrender.OffscreenRenderer(width, height)
    return scene, renderer


def _visual_items(robot, cfg, root):
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.72, 0.77, 0.93, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.45,
    )
    for mesh, pose in robot.visual_trimesh_fk(cfg=cfg).items():
        render_mesh = mesh
        if len(render_mesh.faces) > 6000:
            try:
                render_mesh = render_mesh.simplify_quadric_decimation(face_count=6000)
            except ModuleNotFoundError:
                pass
        yield pyrender.Mesh.from_trimesh(render_mesh, material=material, smooth=False), root @ pose


def render_mesh_video(opt_path, urdf_path, output, fps, stride, max_frames, root_mode, width, height, source_order):
    robot = URDF.load(urdf_path)
    dof_pos, wrist_pos, wrist_rot = _load_motion(opt_path)
    target_joint_names = [joint.name for joint in robot.actuated_joints]
    if source_order == "isaac":
        source_joint_names = _load_isaac_dof_names(urdf_path)
    elif source_order == "urdf":
        source_joint_names = target_joint_names
    else:
        raise ValueError(f"Unknown source_order: {source_order}")
    if dof_pos.shape[1] != len(source_joint_names):
        raise ValueError(f"Motion has {dof_pos.shape[1]} dofs, source order has {len(source_joint_names)} joints")

    frame_ids = np.arange(0, len(dof_pos), stride)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    # Camera is fixed from a representative local-pose bounding box, so playback does not drift.
    mid = int(frame_ids[len(frame_ids) // 2])
    mid_cfg = _make_cfg(source_joint_names, target_joint_names, dof_pos[mid])
    mid_root = _root_transform(wrist_pos[mid], wrist_rot[mid], root_mode)
    bounds = []
    for mesh, pose in robot.visual_trimesh_fk(cfg=mid_cfg).items():
        verts = trimesh.transform_points(mesh.vertices, mid_root @ pose)
        bounds.append(verts)
    all_points = np.concatenate(bounds, axis=0)
    center = all_points.mean(axis=0)
    radius = max(float(np.ptp(all_points, axis=0).max()), 0.16)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sharpa_mesh_frames_") as frame_dir:
        renderer = None
        try:
            scene, renderer = _make_scene(width, height, center, radius)
            for out_idx, frame_id in enumerate(frame_ids):
                cfg = _make_cfg(source_joint_names, target_joint_names, dof_pos[frame_id])
                root = _root_transform(wrist_pos[frame_id], wrist_rot[frame_id], root_mode)
                nodes = []
                for mesh, pose in _visual_items(robot, cfg, root):
                    nodes.append(scene.add(mesh, pose=pose))
                color, _ = renderer.render(scene)
                Image.fromarray(color).save(Path(frame_dir) / f"{out_idx:06d}.png")
                for node in nodes:
                    scene.remove_node(node)
        finally:
            if renderer is not None:
                renderer.delete()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(frame_dir) / "%06d.png"),
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"Wrote {output} ({len(frame_ids)} frames, root_mode={root_mode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True, help="Path to mano2dexhand opt.pkl")
    parser.add_argument("--urdf", required=True, help="Sharpa URDF path")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--root-mode", choices=["local", "rotation", "world"], default="local")
    parser.add_argument("--source-order", choices=["isaac", "urdf"], default="isaac")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    render_mesh_video(
        args.opt,
        args.urdf,
        args.output,
        args.fps,
        args.stride,
        args.max_frames,
        args.root_mode,
        args.width,
        args.height,
        args.source_order,
    )


if __name__ == "__main__":
    main()
