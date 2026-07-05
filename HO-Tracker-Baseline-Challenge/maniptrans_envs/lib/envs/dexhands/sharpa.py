from abc import ABC

import numpy as np

from main.dataset.transform import aa_to_rotmat

from .base import DexHand
from .decorators import register_dexhand


class Sharpa(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self.name = "sharpa"
        self.self_collision = True

        self.body_names = [
            "hand_C_MC",
            "thumb_MC",
            "thumb_PP",
            "thumb_DP",
            "thumb_elastomer",
            "thumb_fingertip",
            "index_PP",
            "index_MP",
            "index_DP",
            "index_elastomer",
            "index_fingertip",
            "middle_PP",
            "middle_MP",
            "middle_DP",
            "middle_elastomer",
            "middle_fingertip",
            "ring_PP",
            "ring_MP",
            "ring_DP",
            "ring_elastomer",
            "ring_fingertip",
            "pinky_MC",
            "pinky_PP",
            "pinky_MP",
            "pinky_DP",
            "pinky_elastomer",
            "pinky_fingertip",
        ]
        self.dof_names = [
            "index_MCP_FE",
            "index_MCP_AA",
            "index_PIP",
            "index_DIP",
            "middle_MCP_FE",
            "middle_MCP_AA",
            "middle_PIP",
            "middle_DIP",
            "pinky_CMC",
            "pinky_MCP_FE",
            "pinky_MCP_AA",
            "pinky_PIP",
            "pinky_DIP",
            "ring_MCP_FE",
            "ring_MCP_AA",
            "ring_PIP",
            "ring_DIP",
            "thumb_CMC_FE",
            "thumb_CMC_AA",
            "thumb_MCP_FE",
            "thumb_MCP_AA",
            "thumb_IP",
        ]

        self.hand2dex_mapping = {
            "wrist": ["hand_C_MC"],
            "thumb_proximal": ["thumb_MC"],
            "thumb_intermediate": ["thumb_PP"],
            "thumb_distal": ["thumb_DP", "thumb_elastomer"],
            "thumb_tip": ["thumb_fingertip"],
            "index_proximal": ["index_PP"],
            "index_intermediate": ["index_MP"],
            "index_distal": ["index_DP", "index_elastomer"],
            "index_tip": ["index_fingertip"],
            "middle_proximal": ["middle_PP"],
            "middle_intermediate": ["middle_MP"],
            "middle_distal": ["middle_DP", "middle_elastomer"],
            "middle_tip": ["middle_fingertip"],
            "ring_proximal": ["ring_PP"],
            "ring_intermediate": ["ring_MP"],
            "ring_distal": ["ring_DP", "ring_elastomer"],
            "ring_tip": ["ring_fingertip"],
            "pinky_proximal": ["pinky_MC", "pinky_PP"],
            "pinky_intermediate": ["pinky_MP"],
            "pinky_distal": ["pinky_DP", "pinky_elastomer"],
            "pinky_tip": ["pinky_fingertip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)

        self.contact_body_names = [
            "thumb_elastomer",
            "index_elastomer",
            "middle_elastomer",
            "ring_elastomer",
            "pinky_elastomer",
        ]
        self.bone_links = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 5],
            [0, 6],
            [6, 7],
            [7, 8],
            [8, 10],
            [0, 11],
            [11, 12],
            [12, 13],
            [13, 15],
            [0, 16],
            [16, 17],
            [17, 18],
            [18, 20],
            [0, 21],
            [21, 22],
            [22, 23],
            [23, 24],
            [24, 26],
        ]
        self.weight_idx = {
            "thumb_tip": [5],
            "index_tip": [10],
            "middle_tip": [15],
            "ring_tip": [20],
            "pinky_tip": [26],
            "level_1_joints": [1, 6, 11, 16, 21],
            "level_2_joints": [2, 3, 7, 8, 12, 13, 17, 18, 22, 23, 24],
        }

        self.Kp_rot = 0.5
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 20
        self.Ki_pos = 0.005
        self.Kd_pos = 0.1

    def __str__(self):
        return self.name


@register_dexhand("sharpa_rh")
class SharpaRH(Sharpa):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/sharpa_wave/sharpa_wave_right.urdf"
        self.side = "rh"
        self.body_names = ["right_" + name for name in self.body_names]
        self.dof_names = ["right_" + name for name in self.dof_names]
        self.relative_rotation = (
            aa_to_rotmat(np.array([-np.pi / 36, 0, 0]))
            @ aa_to_rotmat(np.array([0, 0, np.pi / 36]))
            @ aa_to_rotmat(np.array([0, 0, -np.pi / 2]))
            @ aa_to_rotmat(np.array([0, np.pi, 0]))
        )
        self.hand2dex_mapping = {k: ["right_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["right_" + name for name in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("sharpa_lh")
class SharpaLH(Sharpa):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/sharpa_wave/sharpa_wave_left.urdf"
        self.side = "lh"
        self.body_names = ["left_" + name for name in self.body_names]
        self.dof_names = ["left_" + name for name in self.dof_names]
        self.relative_rotation = (
            aa_to_rotmat(np.array([-np.pi / 36, 0, 0]))
            @ aa_to_rotmat(np.array([0, 0, -np.pi / 36]))
            @ aa_to_rotmat(np.array([0, 0, np.pi / 2]))
        )
        self.hand2dex_mapping = {k: ["left_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["left_" + name for name in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_lh"
