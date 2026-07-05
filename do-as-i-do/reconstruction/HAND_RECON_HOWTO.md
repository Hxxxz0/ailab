# 手部重建(任务二)运行手册 — HaWoR + headless 叠加

本文档记录如何用 HaWoR 从一段手物视频重建手部(MANO),并在**无显示器 / 无 OpenGL 的服务器**上生成叠加视频。以 `grasp_drink_yykx / camera_side_1` 为例,全程在 Jensen conda 的 `hawor` / `sam3d` 环境里跑。

---

## 0. 前置

- conda:`source /mnt/data/cpfs/Jensen/miniconda3/etc/profile.d/conda.sh`
- 环境:`hawor`(跑 HaWoR)、`sam3d`(跑 headless 叠加,自带 pytorch3d)
- 输入视频:`.../sam3_tests/track_side1/camera_side_1.mp4`(1280×720,265 帧)
- GPU:用空闲卡,示例用 `CUDA_VISIBLE_DEVICES=1`

---

## 1. 放置 MANO 模型(一次性)

HaWoR 需要许可下载的 MANO 权重。从 `do-as-i-do/mano_v1_2.zip` 取出左右手 pkl,放进 HaWoR 的两个 MODEL_PATH:

- 右手 → `modules/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl`
- 左手 → `modules/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl`

```bash
cd /mnt/workspace/Jensen/project/ailab/do-as-i-do
TMP=$(mktemp -d)
unzip -j -o mano_v1_2.zip \
    "mano_v1_2/models/MANO_RIGHT.pkl" "mano_v1_2/models/MANO_LEFT.pkl" -d "$TMP"
H=/mnt/workspace/Jensen/project/ailab/do-as-i-do/reconstruction/modules/HaWoR
cp "$TMP/MANO_RIGHT.pkl" "$H/_DATA/data/mano/MANO_RIGHT.pkl"
cp "$TMP/MANO_LEFT.pkl"  "$H/_DATA/data_left/mano_left/MANO_LEFT.pkl"
rm -rf "$TMP"
```

> 路径由 `hawor/utils/process.py` 里的 `MODEL_PATH`(右 `_DATA/data/mano`、左 `_DATA/data_left/mano_left`)决定;smplx `MANOLayer` 按 `is_rhand` 自动找 `MANO_RIGHT.pkl` / `MANO_LEFT.pkl`。
> `_DATA/data/mano_mean_params.npz` 与权重(`weights/hawor/checkpoints/hawor.ckpt`、`infiller.pt`、`weights/external/detector.pt`、`model_config.yaml`)在环境配置阶段已就位。

---

## 2. 确定焦距 `--img_focal`

`demo.py` 无 `--img_focal` 时回退默认 600(1280×720 偏小)。用物体追踪 `layout.json` 里 MoGe 估计的归一化内参换算更准:

- `fx_norm = 0.7203`,`fy_norm = 1.2805`,`cx=cy=0.5`
- `focal_px = fx_norm × W = 0.7203 × 1280 ≈ 922`(`fy_norm × H = 1.2805 × 720 ≈ 922`,一致)

→ 本视频用 **`--img_focal 922`**。换视频时同法从对应 layout 或自行标定重算。

---

## 3. 跑 HaWoR 手部重建

必须在 `modules/HaWoR` 目录下运行(权重和 `_DATA` 用的是相对路径);`--static_camera` 走静态相机分支,**不需要 DROID-SLAM/lietorch**。

```bash
source /mnt/data/cpfs/Jensen/miniconda3/etc/profile.d/conda.sh
conda activate hawor
cd /mnt/workspace/Jensen/project/ailab/do-as-i-do/reconstruction/modules/HaWoR

VID=/mnt/workspace/Jensen/project/ailab/do-as-i-do/reconstruction/sam3_tests/track_side1/camera_side_1.mp4

CUDA_VISIBLE_DEVICES=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python demo.py \
    --video_path "$VID" \
    --vis_mode cam \
    --img_focal 922 \
    --static_camera
```

流程:抽帧 → 手部检测/跟踪 → 运动估计 → infiller → 保存手部数据 →(最后一步渲染叠加)。

### 产物 —— 手部数据(核心,任务二交付物)

`camera_side_1/all_hand_meshes.npz`,均为 **相机系**,265 帧:

| key | shape | 含义 |
|---|---|---|
| `left_vertices` / `right_vertices` | (265, 778, 3) | 手部网格顶点 |
| `left_faces` / `right_faces` | (1552, 3) | 面片 |
| `left_joints` / `right_joints` | (265, 21, 3) | 21 关节 |
| `{left,right}_trans` | (265, 3) | 全局平移 |
| `{left,right}_rot` | (265, 3) | 全局旋转(axis-angle) |
| `{left,right}_hand_pose` | (265, 45) | MANO 手指姿态 |
| `{left,right}_betas` | (265, 10) | MANO 形状 |
| `{left,right}_valid` | (265,) bool | 每帧该手是否有效 |

---

## 4. ⚠️ 官方叠加视频在无头服务器上会失败

`demo.py --vis_mode cam` 最后用 **aitviewer / moderngl EGL** 渲染 `overlay.mp4`,在无显示、无 GL 上下文的服务器上报:

```
ValueError: Requested OpenGL version 450, got version 0
run_vis2.py → viewer_utils.ARCTICViewer → HeadlessRenderer → moderngl.create_standalone_context
```

**注意:此时 `all_hand_meshes.npz` 已经保存**(它在渲染步骤之前写),所以**手部重建本身是成功的**,只是那段可视化视频没出来。用下面的 headless 方案补出视频。

---

## 5. Headless 叠加视频(pytorch3d 方案,推荐)

用 `scripts/render_hands_overlay.py`(本项目新增)在 `sam3d` 环境用 **pytorch3d** 渲染,不依赖 OpenGL。npz 顶点已是相机系(x-right, y-down, z-fwd),脚本内部做 `cam_to_pytorch3d` 后按焦距投影,再和视频帧 alpha 合成,ffmpeg 出 mp4。

```bash
source /mnt/data/cpfs/Jensen/miniconda3/etc/profile.d/conda.sh
conda activate sam3d
cd /mnt/workspace/Jensen/project/ailab/do-as-i-do/reconstruction/sam3_tests/track_side1

CUDA_VISIBLE_DEVICES=1 python \
    /mnt/workspace/Jensen/project/ailab/do-as-i-do/reconstruction/scripts/render_hands_overlay.py \
    --video  camera_side_1.mp4 \
    --npz    camera_side_1/all_hand_meshes.npz \
    --output camera_side_1/hand_overlay.mp4 \
    --focal  922
```

参数:
- `--focal`(必须和第 2 步一致);`--cx/--cy` 默认取画面中心 W/2、H/2
- `--alpha` 叠加不透明度(默认 0.75);`--fps` 输出帧率(默认 30)
- 右手渲染成蓝色、左手橙色;每帧按 `{left,right}_valid` 跳过无效手

### 产物 —— 叠加视频

`camera_side_1/hand_overlay.mp4`(265 帧,MANO 手网格叠在原视频上)。

---

## 6. 快速自检

```bash
conda activate sam3d
cd .../sam3_tests/track_side1
# 看 npz 字段
python3 -c "import numpy as np; d=np.load('camera_side_1/all_hand_meshes.npz'); [print(k,d[k].shape) for k in d.files]"
# 抽 4 帧拼图确认贴合
python3 - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('camera_side_1/hand_overlay.mp4'); t=[]
for i in [30,110,180,240]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=cap.read()
    if ok: t.append(cv2.resize(f,(480,270)))
cv2.imwrite('camera_side_1/_hand_overlay_grid.png',
            np.vstack([np.hstack(t[:2]),np.hstack(t[2:])]))
PY
```

---

## 7. 换一个视频要改什么

1. `--video_path` 指向新 mp4(第 3 步),仍在 `modules/HaWoR` 下运行。
2. 重算 `--img_focal`(第 2 步:归一化内参 × 分辨率,或用真实标定)。
3. 叠加时 `--video` / `--npz` / `--output` 换成新视频对应路径,`--focal` 保持一致。

## 关键文件

- `modules/HaWoR/demo.py` —— HaWoR 主程序(已打补丁:`hawor_slam` 惰性导入,static_camera 不需 lietorch)
- `scripts/render_hands_overlay.py` —— 无头 pytorch3d 手部叠加(新增)
- `_DATA/data/mano/MANO_RIGHT.pkl`、`_DATA/data_left/mano_left/MANO_LEFT.pkl` —— MANO 权重
