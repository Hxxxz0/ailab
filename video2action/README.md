# video2action

从手-物演示视频重建、重定向到机器人手、再到部署的完整管线。每个阶段各占一个子目录。

- **`reconstruction/`** — 从手-物演示视频重建物体 + 手部并做 6-DoF 位姿追踪
  （SAM3 → SAM-3D mesh → MoGe pointmaps → HaWoR → TAPIR → guided diffusion 追踪 → 投影）。
  详见 [`reconstruction/README.md`](reconstruction/README.md)。对应赛题**任务二/三**。
- **`retargeting/`** — 把重建出的手-物演示重定向到机器人手
  （数据处理 → 凸分解 → MJCF 场景生成 → IK → MuJoCo Warp 采样式 MPC），直接消费 reconstruction 的输出。
  详见 [`retargeting/README.md`](retargeting/README.md)。
- **`deployment/`** — 把重定向结果在真实机器人上回放：MuJoCo 回放/IK 生成双 UR3e 关节轨迹，
  再下发到 UR3e 机械臂 + Sharpa Wave 灵巧手。详见 [`deployment/README.md`](deployment/README.md)。

> 上游 fork 子模块位于 `reconstruction/modules/`（未随本提交包含，见根目录 README 的许可与 upstream 表）。
