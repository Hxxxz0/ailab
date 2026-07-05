# ailab — 从人类演示视频到灵巧手操作

从单目/多视角**人类操作演示视频**出发，重建**手部**与**被操作物体**的三维形状与
6-DoF 轨迹，重定向到 **Sharpa 灵巧手**，并在 **IsaacGym** 中训练/追踪与评测。
覆盖赛题任务一 ~ 任务四。

> 本仓库为**提交用源码快照**：仅含第一方代码（约 3.4 MB / 230+ 文件）。
> 模型权重、数据集、运行产物（outputs）、大体积网格/视频等二进制、以及
> `video2action/reconstruction/modules/` 下的上游 fork 子模块（SAM3 / SAM-3D-Objects /
> HaWoR / Fast-SAM3D / tapnet，各自独立许可）**均未包含**，请按下文各自 upstream 获取。

---

## 目录结构

```
ailab/
├── video2action/          # 手/物重建 + 重定向 + 部署管线
│   ├── reconstruction/    #   任务二/三：手部(HaWoR/MANO) + 物体(SAM3/SAM-3D) 重建与 6-DoF 追踪
│   │   ├── scripts/       #     第一方管线脚本（hand/ object/ 及公共 stage 脚本）
│   │   ├── run_hand_task2.sh      / run_object_task3.sh   # 全数据集多卡批处理入口
│   │   ├── config/        #     路径与参数
│   │   └── modules/       #     上游子模块占位（未包含，见下方 upstream）
│   ├── retargeting/       #   MANO → 灵巧手 重定向
│   └── deployment/        #   mujoco 回放/部署
│
└── dexhand_baseline/      # 任务一/四：Sharpa 灵巧手 RL 追踪 baseline（原 HO-Tracker-Baseline-Challenge）
    ├── main/              #   dataset(mano2dexhand) / rl(train, render_rollout) / eval_score
    ├── maniptrans_envs/   #   IsaacGym 任务与机器人 asset（urdf/mjcf；大网格二进制未含）
    ├── lib/  DexManipNet/ scripts/
    └── setup.py
```

## 任务对照

| 任务 | 内容 | 代码位置 |
| --- | --- | --- |
| 任务一 | Sharpa 重定向与 baseline 数据契约 | `dexhand_baseline/main/dataset/mano2dexhand.py`、`dexhand_baseline/main/eval_score.py` |
| 任务二 | 手部重建与轨迹恢复（MANO） | `video2action/reconstruction/scripts/hand/`、`run_hand_task2.sh` |
| 任务三 | 物体形状重建 + IsaacGym asset | `video2action/reconstruction/scripts/object/`、`run_object_task3.sh` |
| 任务四 | 综合：接口转化 + 训练/追踪 + 成败分析 | `dexhand_baseline/main/rl/`（train / render_rollout / rollout / score） |

---

## 环境依赖

- **GPU**：≥ 32 GB 显存的 NVIDIA GPU。
- **IsaacGym Preview 4**（任务一/四仿真；`dexhand_baseline` 的 `maniptrans` conda env）。
- **重建侧 conda envs**：`sam3`（分割）、`sam3d`（mesh/追踪/导出）、`hawor`（手）、`tapnet`（光流）。
- **HuggingFace 权限**：`facebook/sam-3d-objects`、`facebook/sam3`；HaWoR 需 MANO 授权
  （https://mano.is.tue.mpg.de ，**许可受限，不随本仓库分发**）。

## 复现

```bash
# —— 任务二：全数据集多卡手部重建 ——
cd video2action/reconstruction
./run_hand_task2.sh 0,1,2,3,4,5,6

# —— 任务三：全数据集多卡物体重建 + IsaacGym 资产 ——
./run_object_task3.sh 0,1,2,3,4,5

# —— 任务一/四：重定向 → 训练 → rollout → 评分 ——
cd ../../dexhand_baseline            # conda activate maniptrans; export PYTHONPATH=.
python main/dataset/mano2dexhand.py --data_idx pipettepress@0 --side right --dexhand sharpa --headless
python main/rl/train.py task=ResDexHand dexhand=sharpa side=RH headless=true dataIndices=[pipettepress@0]
python main/eval_score.py --tag dump_ppress_cbest_ --dexhand sharpa
```

产物默认写入各自的 `outputs/`（未随仓库提交）。

---

## 上游子模块（未包含，需自行获取）

`video2action/reconstruction/modules/` 下为带本地改动的上游 fork，各自保留原许可，请分别克隆：

| 子模块 | upstream | 许可 |
| --- | --- | --- |
| sam-3d-objects | facebookresearch/sam-3d-objects | SAM License (Meta) |
| sam3 | facebookresearch/sam3 | SAM License (Meta) |
| Fast-SAM3D | wlfeng0509/Fast-SAM3D | MIT |
| HaWoR | ThunderVVV/HaWoR | CC BY-NC-ND 4.0 |
| tapnet | google-deepmind/tapnet | Apache-2.0 |

## 未包含内容

模型权重（`*.ckpt/*.pt/*.pth`）、数据集与 GT、运行产物 `outputs/`、
大体积网格与媒体（`*.obj/*.STL/*.glb/*.mp4/*.png`）、MANO 许可受限文件、
IsaacGym 安装包，均出于体积与许可考虑排除；本仓库聚焦可读的第一方源码。
