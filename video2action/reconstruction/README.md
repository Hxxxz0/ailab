# video2action · Reconstruction

Hand and object **reconstruction + 6-DoF pose tracking** from a single hand-object demo video. Given a
video, a reference frame, an object name, and the anchor hand, the pipeline segments the object
and hand, reconstructs a 3D mesh for the object, estimates per-frame object pose, and reconstructs the hand (HaWoR).

## Layout

```
reconstruction/
├── run_pipeline.sh          # the driver 
├── config/paths.sh          # ← the ONLY file you edit to relocate things
├── scripts/                 # 7 standalone first-party stage scripts
├── modules/             # submodules with our changes
├── weights/                 # model weights 
├── env/                     # conda env docs (see env/README.md)
└── setup/                   # 00 submodules · 01 envs · 02 weights
```

## Requirements
- An NVIDIA GPU with ≥ 32 GB VRAM.
- HuggingFace auth with access to the repos `facebook/sam-3d-objects` and `facebook/sam3`; Plus a MANO download
  (https://mano.is.tue.mpg.de) for HaWoR.
- SAM 3 segmentation is implemented with a click-based GUI needing an X display
  (`config/paths.sh` sets `SAM3_DISPLAY=:1`; on a headless host, use forwarding or try text based prompting).

## Setup (one time)

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/malik-group/video2action.git
cd video2action/reconstruction
./setup/00_init_submodules.sh                 # only needed if you didn't clone with recursive submodules
./setup/01_create_envs.sh                     # FALLBACK build of all 4 envs (sam3, sam3d, hawor, tapnet) — prefer each fork's own setup, see "Setting up the conda envs" below
./setup/02_fetch_weights.sh --download        # fetch weights (needs hf auth)
```
**Setting up the conda envs.** The recommended route is to build each env by following
its fork's own setup instructions (the repos vendored under `modules/`):

- `sam3`  → [malik-group/sam3](https://github.com/malik-group/sam3) (`modules/sam3`)
- `sam3d` → [malik-group/sam-3d-objects](https://github.com/malik-group/sam-3d-objects) (`modules/sam-3d-objects`, see its `doc/setup.md`) 
- `hawor` → [malik-group/HaWoR](https://github.com/malik-group/HaWoR) (`modules/HaWoR`)
- `tapnet` → [malik-group/tapnet](https://github.com/malik-group/tapnet) (`modules/tapnet`)

See [`env/README.md`](env/README.md) for the per-env cu128 recipes (or `./setup/01_create_envs.sh` to script them).

After the `sam3d` env is built, two manual Stage-2 steps are needed: un-shadow the repo's
`notebook/` package (`pip uninstall -y notebook`) and build the Mip-Splatting
`diff_gaussian_rasterization` for the renderer's `inria` backend. Commands in
[`env/README.md`](env/README.md).

Review `config/paths.sh`.


## Run

```bash
./run_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# e.g.
./run_pipeline.sh whisking/whisking.mp4 125 whisk right
```

### Task-2 batch hand reconstruction (whole dataset, multi-GPU)

Reconstruct hands (MANO + 2D mask + de-jittered trajectory + QC) for **every
(sequence, camera) clip** of the HO-Tracker `human_demo` dataset, distributed across
GPUs. Real camera intrinsics are read from each `camera_calib/<cam>/cam_intr.pkl`.

```bash
./run_hand_task2.sh 0,1,2,3,4,5,6            # all 36 clips across 7 GPUs
./run_hand_task2.sh 0 --sequences <seq_dir> --cameras camera_side_1   # one clip
./run_hand_task2.sh 0,1,2 --skip-existing    # resume, skip finished clips
```

Driver `scripts/hand/run_hand_batch.py` runs, per clip: HaWoR (`hawor` env) →
`smooth_hand_npz.py` → `render_hands_overlay.py` → `export_hand_mask.py` →
`hand_qc.py` (all `sam3d` env). Modules live in `scripts/hand/`.

**Outputs** (`outputs/hand/<sequence>/<camera>/`, camera frame — Task-4 ready):
`all_hand_meshes[_smoothed].npz`, `hand_mesh_overlay.mp4`, `hand_mask_overlay.mp4`,
`keyframes_grid.png`, `qc_report.json`, plus `intrinsics.json` / `img_focal.txt` /
`config.json`. Batch index: `outputs/hand/INDEX.md` + `SUMMARY.json`. Detailed
single-clip walkthrough: [`HAND_RECON_HOWTO.md`](HAND_RECON_HOWTO.md).

> Headless note: HaWoR's aitviewer overlay needs OpenGL and fails on this host; the npz/masks
> are saved *before* that step, and `render_hands_overlay.py` re-renders the overlay with PyTorch3D.

### Task-3 batch object reconstruction + IsaacGym asset (whole dataset, multi-GPU)

Reconstruct each manipulated object (mesh + 2D mask + de-jittered 6-DoF trajectory +
**IsaacGym-loadable URDF asset** + QC) for every `(sequence, object)` of the `human_demo`
dataset. One asset per object; multi-object sequences (pipette + beaker + test tube) each
get their own.

```bash
./run_object_task3.sh 0,1,2,3,4,5                    # whole dataset across 6 GPUs
./run_object_task3.sh 0 --sequences <seq_dir>        # one sequence
```

Driver `scripts/object/run_object_batch.py`, two phases: **(1) track** per (seq, object,
camera) via `scripts/object/track_view.sh` (frames → SAM3 seg → SAM3D mesh → keyframe
diffusion track → **phase-aware upright pose-lock** → project); **(2) finalize** per object:
`select_view.py` → `build_asset.py` + `validate_isaacgym.py` (**maniptrans** env: CoACD
collision + URDF + IsaacGym headless load/settle) → `mesh_gallery.py` → `export_pose.py`
(camera-frame `object_pose.pkl` + Task-1 baseline mirror) → `object_qc.py` → `REVIEW/`.

- **Pose-lock** (`track_keyframes.py`): objects stay vertical per each `video/1.txt` note, so
  the object axis (swing) is locked upright during the noted vertical intervals — kills the
  rotation wobble (rotation jitter ↓ ~450×). Config in `run_object_batch.py:SEQUENCE_LOCK_SEC`.
- **Segmentation**: SAM3 text candidate lists (`bottle|a bottle|...`) for bottle/bread; for the
  thin pipette / blue-white glassware use the **web point-picker** first:
  ```bash
  conda activate sam3d && python scripts/object/point_picker.py --port 8891   # click objects; saves point_seeds.json
  ```
  (Run it in a VSCode terminal — the port auto-forwards; or open via DSW `/proxy/8891/`.)

**Outputs** (`outputs/object/<sequence>/`, Task-1/4 ready): `object_asset/` (visual+collision
+`object.urdf`+`object_pose.pkl`+`isaacgym_validation/`), `mask/`, `mesh_gallery/`, `tracking/`,
`baseline_export/{side}_urdf/` + `{side}_obj.pkl`, `qc_report.json`, `REVIEW/`. Index:
`outputs/object/INDEX.md` + `SUMMARY.json`. Walkthrough: [`OBJECT_ASSET_HOWTO.md`](OBJECT_ASSET_HOWTO.md).

### Details on Pipeline Stages
| # | stage | script | env |
|---|---|---|---|
| 0 | extract frames | ffmpeg | sam3 |
| 1 | SAM3 segmentation (object click + hand text) | `scripts/run_sam3_video.py` | sam3 |
| 2 | masks → 3D mesh | `modules/sam-3d-objects/generate_mesh_sam3d.py` | sam3d |
| 2 | MoGe pointmaps (reference frame + all frames) | `scripts/get_pointmap_dir.py` | sam3d |
| 2 | hand reconstruction | `modules/HaWoR/demo.py` | hawor |
| 2 | gravity estimation (GeoCalib) | `scripts/predict_video_gravity.py` | sam3d |
| 2.5 | velocity tracking | `scripts/tapir_velocity_tracking.py` | tapnet |
| 3 | guided pose prediction | `modules/Fast-SAM3D/track_object.py` | sam3d |
| 3 | project mesh / to camera frame | `scripts/run_project_mesh_combined.py`, `scripts/convert_layout_to_camera_frame.py` | sam3d |
| 4 | optimize translation/scale | `scripts/optimize_translation_scale.py` | sam3d |
| 4 | (optional) 3D viser viz | `scripts/visualize_3d.py` | sam3d |


### Outputs (under the video's directory)
`video_segmentation/masks/…`, per-object `.obj` meshes, `*_pointmap.npy` / `*_intrinsics.txt`,
HaWoR `all_hand_meshes.npz`, `gravity.json` (camera-frame up direction from GeoCalib),
and `obj_tracking_out/<object>/combined_visualization/`
with `layout.json → layout_camera_frame.json → layout_camera_frame_optimized.json` and projected
frames. This directory is the input to the [`retargeting/`](../retargeting/README.md) pipeline.


### Visualize the processed output yourself

To explore the processed `whisk` demo (`whisking/`) in an interactive 3D (viser) viewer
without re-running the pipeline, first extract the video frames. Use the **same** ffmpeg
invocation as the pipeline's Step 0 so the frame numbering matches the tracking output:

```bash
# from the reconstruction/ directory — same flags as run_pipeline.sh Step 0
VIDEO_DIR=whisking
mkdir -p "$VIDEO_DIR/all_frames"
ffmpeg -i "$VIDEO_DIR/whisking.mp4" -vsync 0 -start_number 0 "$VIDEO_DIR/all_frames/%06d.png"
```

Then launch the viewer:

```bash
conda activate sam3d
VIDEO_DIR=whisking
OBJECT_ID=whisk
LAYOUT_JSON_OPT="$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame_optimized.json"
python scripts/visualize_3d.py \
    --frames-dir "$VIDEO_DIR/all_frames" \
    --layout-json "$LAYOUT_JSON_OPT" \
    --mesh "$VIDEO_DIR/video_segmentation/masks/frame_000125_masks/$OBJECT_ID/$OBJECT_ID.obj" \
    --hand-meshes "$VIDEO_DIR/whisking/all_hand_meshes.npz" \
    --scale 0.1808 \
    --translation-scale 1.0 \
    --hands both \
    --port 8080
```


## Credits & licenses
The `modules/` contain external references with our changes to the sources, and each of them retain their upstream `LICENSE`. We gratefully
acknowledge the original authors.

| fork | upstream @ pinned commit | fork commit | license |
|---|---|---|---|
| `malik-group/sam-3d-objects` | facebookresearch/sam-3d-objects @ `81a8237` | `875b010` | SAM License (Meta) |
| `malik-group/Fast-SAM3D`     | wlfeng0509/Fast-SAM3D @ `c0f99e8`           | `823d478` | MIT (+ embedded SAM-3D under SAM License) |
| `malik-group/HaWoR`          | ThunderVVV/HaWoR @ `de90272`                | `2c3fa0c` | CC BY-NC-ND 4.0 |
| `malik-group/tapnet`         | google-deepmind/tapnet @ `96d3f84`          | `f2f8888` | Apache-2.0 |
| `malik-group/sam3`           | facebookresearch/sam3 @ `757bbb0`           | `b8e18f5` | SAM License (Meta) |

