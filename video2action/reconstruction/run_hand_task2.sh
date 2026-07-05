#!/bin/bash
# ═══════════════════════ 任务二 批量手部重建入口 ═══════════════════════
# 处理 HO-Tracker human_demo 全部 (sequence, camera) clip:
#   HaWoR(hawor env) -> 去抖/3D叠加/2D mask/QC(sam3d env),多 GPU 并行。
# 用法: ./run_hand_task2.sh [GPUS] [EXTRA ARGS...]
#   ./run_hand_task2.sh 0,1,2,3,4,5,6
#   ./run_hand_task2.sh 1 --limit 1                       # 单 clip 冒烟
#   ./run_hand_task2.sh 0,1,2 --sequences weigh_bread__2026_0701_0044_30
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/Jensen/project/ailab/HO-Tracker-data/human_demo}"
OUT="${OUT:-$HERE/outputs/hand}"
GPUS="${1:-0,1,2,3,4,5,6}"; shift || true

source /mnt/data/cpfs/Jensen/miniconda3/etc/profile.d/conda.sh
conda activate sam3d   # driver runs in sam3d; per-step it switches envs itself

python "$HERE/scripts/hand/run_hand_batch.py" \
    --data-root "$DATA_ROOT" \
    --out "$OUT" \
    --gpus "$GPUS" \
    "$@"
