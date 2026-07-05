# Reconstruction env setup — notes (A800/A100, aliyun, Jensen conda)

Environment configured for the video2action **reconstruction** pipeline (challenge tasks 2 & 3)
on this host. All 4 conda envs live under the **Jensen conda** and persist on CPFS:
`/mnt/data/cpfs/Jensen/miniconda3/envs/{sam3,sam3d,hawor,tapnet}`.

Build scripts + logs: `reconstruction/_setup_logs/`.

## Status — DONE & verified

| env | torch | verified |
|---|---|---|
| `sam3`   | 2.8.0+cu128 | `import sam3` + **SAM3 predictor loads** from local ModelScope copy (offline) |
| `sam3d`  | 2.5.1+cu121 | pytorch3d/kaolin/geocalib import + **full SAM3D pipeline + all checkpoints load** |
| `hawor`  | 2.7.0+cu128 | `demo.py --help` imports clean (lazy-SLAM patch) |
| `tapnet` | 2.7.0+cu128 | `import tapnet`, cuda ok |

Weights present (`weights/` + `modules/*/checkpoints/hf/`, HaWoR `weights/`): SAM3D (all
`pipeline.yaml` ckpts), SAM3 (`sam3.pt`+config), HaWoR (`hawor.ckpt`/`infiller.pt`/`detector.pt`/`model_config.yaml`),
TAPIR (`bootstapir_checkpoint_v2.pt`).

## How to run

```bash
source /mnt/data/cpfs/Jensen/miniconda3/etc/profile.d/conda.sh   # use the Jensen conda
cd /mnt/workspace/Jensen/project/ailab/video2action/reconstruction
./run_pipeline.sh whisking/whisking.mp4 125 whisk right
```
`config/paths.sh` is already correct (env names, `CUDA_VISIBLE_DEVICES=0` — GPU7 is busy,
`TAPNET_CKPT`, `SAM3_DISPLAY=:1`).

## Fully headless — no display needed

`run_pipeline.sh` was patched so **SAM3 stage-1 needs no X/VNC**: the object is segmented with a
SAM3 **text prompt** (`--text "$OBJ_NAME"`) instead of the `--click` GUI, and the hand already used
text. Verified headless (DISPLAY unset) on the whisking sample: text "whisk" → score 0.77, all 172
frames tracked. Overrides:
- `SAM3_OBJ_TEXT="..."` — custom object text (default = the object name arg).
- `SAM3_OBJ_POINTS="x,y"` (+ optional `SAM3_OBJ_POINT_LABELS="1"`) — headless point prompt for
  precise selection: extract the ref frame once (`ffmpeg -i VIDEO -vf "select=eq(n\,N)" -vframes 1 f.png`),
  eyeball a pixel on the object.

## Remaining user-side step

**MANO (required for HaWoR / task 2)** — license-gated manual download from
https://mano.is.tue.mpg.de → place:
- `modules/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl`
- `modules/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl`

## Non-obvious decisions / fixes applied (why)

- **`unset PIP_CONSTRAINT`** in every build — the container sets `PIP_CONSTRAINT=/etc/pip/constraint.txt`
  pinning torch to an NVIDIA build; it breaks all custom torch installs.
- **torch via aliyun find-links** (`-f https://mirrors.aliyun.com/pytorch-wheels/cuXXX/` with exact
  `+cuXXX` pins) — the aliyun mirror is a flat listing, NOT a PEP503 index (so `--index-url` fails).
- **sam3d uses cu121** (fork's native recipe via `environments/default.yml`, which bundles the
  cuda-toolkit 12.1 + gcc 12.4 toolchain) — A800 is sm_80 so cu121 runs fine; kaolin has a prebuilt
  cu121 wheel (no source build). `TORCH_CUDA_ARCH_LIST=8.0` needed or old pytorch3d mis-detects sm_100.
- **flash_attn skipped** in sam3d — default attention backend is `sdpa` (+ xformers present); the fork
  also forces `rendering_engine="pytorch3d"`, so **nvdiffrast is unneeded** too. (flash-attn source
  build stalls fetching a prebuilt wheel; not worth it.)
- **HaWoR: DROID-SLAM/lietorch NOT built** — the pipeline runs `--static_camera`; `demo.py` was patched
  to import `hawor_slam` (→ droid/lietorch) lazily inside the non-static-camera branch. Metric3D and
  `droid.pth` are likewise unneeded on this path.
- **SAM3 gated model** served from local ModelScope copy: `weights/sam3-model/` is symlinked into the
  HF hub cache (`_setup_logs/wire_sam3_cache.sh`); `run_pipeline.sh`'s SAM3 calls run with
  `HF_HUB_OFFLINE=1` (patched). hf-mirror 403s on this gated repo, so offline-from-cache is required.
- ffmpeg: system `ffmpeg` 6.1.1 used (skipped all conda ffmpeg installs).
