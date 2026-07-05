#!/usr/bin/env python
import os
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401
from isaacgym import gymapi

import maniptrans_envs.lib.envs.dexhands  # noqa: F401
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def _urdf_names(path):
    root = ET.parse(path).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    joints = {
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in {"fixed", "floating"}
    }
    meshes = [mesh.attrib.get("filename", "") for mesh in root.findall(".//mesh")]
    return links, joints, meshes


def _isaac_dof_names(path):
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, gymapi.SimParams())
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = False
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_root = os.path.dirname(path)
    asset_file = os.path.basename(path)
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    names = [gym.get_asset_dof_name(asset, i) for i in range(gym.get_asset_dof_count(asset))]
    gym.destroy_sim(sim)
    return names


def _check_side(side):
    hand = DexHandFactory.create_hand("sharpa", side)
    links, joints, meshes = _urdf_names(hand.urdf_path)
    isaac_dof_names = _isaac_dof_names(hand.urdf_path)
    missing_bodies = sorted(set(hand.body_names) - links)
    missing_contacts = sorted(set(hand.contact_body_names) - links)
    missing_dofs = sorted(set(hand.dof_names) - joints)
    bad_mesh_paths = [mesh for mesh in meshes if mesh.startswith("package://")]
    bad_weight_idx = {
        name: idxs
        for name, idxs in hand.weight_idx.items()
        if any(idx < 0 or idx >= hand.n_bodies for idx in idxs)
    }

    assert os.path.exists(hand.urdf_path), hand.urdf_path
    assert hand.n_dofs == 22, hand.n_dofs
    assert not missing_bodies, missing_bodies
    assert not missing_contacts, missing_contacts
    assert not missing_dofs, missing_dofs
    assert not bad_mesh_paths, bad_mesh_paths[:5]
    assert not bad_weight_idx, bad_weight_idx
    assert hand.dof_names == isaac_dof_names, {
        "dexhand": hand.dof_names,
        "isaac": isaac_dof_names,
    }
    assert len(hand.contact_body_names) == 5, hand.contact_body_names
    assert set(hand.dex2hand_mapping) == set(hand.body_names)
    print(f"{side}: {hand} bodies={hand.n_bodies} dofs={hand.n_dofs} urdf={hand.urdf_path}")


def main():
    _check_side("left")
    _check_side("right")
    print("Sharpa interface check passed.")


if __name__ == "__main__":
    main()
