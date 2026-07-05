#!/usr/bin/env python3
"""
Batch Task-2 hand reconstruction over the whole HO-Tracker human_demo dataset.

For every (sequence, camera) clip it runs, pinned to one GPU:
  1) HaWoR demo.py (env `hawor`)      -> all_hand_meshes.npz + tracks_*/model_masks.npy
  2) smooth_hand_npz.py (env `sam3d`) -> all_hand_meshes_smoothed.npz  (de-jitter)
  3) render_hands_overlay.py (sam3d)  -> hand_mesh_overlay.mp4  (real fx,fy,cx,cy)
  4) export_hand_mask.py (sam3d)      -> hand_mask_overlay.mp4 + hand_mask.npy
  5) hand_qc.py (sam3d)               -> qc_report.json + keyframes_grid.png

Clips are distributed across the given GPUs (one worker thread per GPU, each pulls
from a shared queue). Output layout (Task-4 ready, camera frame):
  <out>/<sequence>/<camera>/{input.mp4, intrinsics.json, img_focal.txt, config.json,
                             all_hand_meshes.npz, all_hand_meshes_smoothed.npz,
                             hand_mesh_overlay.mp4, hand_mask_overlay.mp4, hand_mask.npy,
                             keyframes_grid.png, qc_report.json, hawor.log, post.log}
  <out>/SUMMARY.json, <out>/INDEX.md

Usage:
  python run_hand_batch.py --data-root .../HO-Tracker-data/human_demo \
      --out .../reconstruction/outputs/hand --gpus 0,1,2,3,4,5,6
"""
import argparse
import json
import os
import glob
import queue
import subprocess
import threading
import time

from dataset import enumerate_clips
from intrinsics import load_intrinsics

HERE = os.path.dirname(os.path.abspath(__file__))
RECON = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = os.path.join(RECON, "scripts")
HAWOR_DIR = os.path.join(RECON, "modules", "HaWoR")
CONDA_SH = "/mnt/data/cpfs/Jensen/miniconda3/etc/profile.d/conda.sh"

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def run_in_env(env, cmd, gpu, cwd, logfile, extra_env=""):
    """Run `cmd` (a shell string) inside conda env `env`, GPU pinned, tee to logfile."""
    full = (
        f"source {CONDA_SH} && conda activate {env} && "
        f"export CUDA_VISIBLE_DEVICES={gpu} && export TORCH_CUDA_ARCH_LIST=8.0 && "
        f"{extra_env} cd {cwd} && {cmd}"
    )
    with open(logfile, "a") as lf:
        lf.write(f"\n$ [{env} gpu{gpu}] {cmd}\n")
        lf.flush()
        rc = subprocess.run(["bash", "-c", full], stdout=lf, stderr=subprocess.STDOUT).returncode
    return rc


def process_clip(clip, gpu, out_root, args):
    wd = os.path.join(out_root, clip.sequence, clip.camera)
    os.makedirs(wd, exist_ok=True)
    input_mp4 = os.path.join(wd, "input.mp4")
    if not os.path.exists(input_mp4):
        os.symlink(os.path.abspath(clip.video), input_mp4)

    npz = os.path.join(wd, "input", "all_hand_meshes.npz")
    smoothed = os.path.join(wd, "all_hand_meshes_smoothed.npz")
    npz_local = os.path.join(wd, "all_hand_meshes.npz")
    result = {"clip": clip.clip_id, "gpu": gpu, "workdir": wd,
              "object": clip.object_name, "anchor_hand": clip.anchor_hand}

    if args.skip_existing and os.path.exists(os.path.join(wd, "qc_report.json")):
        result["status"] = "skipped"
        return result

    # --- intrinsics + config ---
    intr = load_intrinsics(clip.cam_intr, clip.video)
    with open(os.path.join(wd, "intrinsics.json"), "w") as f:
        json.dump(intr, f, indent=2)
    with open(os.path.join(wd, "img_focal.txt"), "w") as f:
        f.write(str(intr["fx"]))
    with open(os.path.join(wd, "config.json"), "w") as f:
        json.dump({"frame_number": 0, "object_names": [clip.object_name],
                   "anchor_hand": clip.anchor_hand}, f, indent=2)

    hawor_log = os.path.join(wd, "hawor.log")
    post_log = os.path.join(wd, "post.log")
    t0 = time.time()

    # --- 1) HaWoR (aitviewer/OpenGL step fails headless AFTER npz is saved -> ignore rc) ---
    run_in_env(
        "hawor",
        f"python demo.py --video_path {input_mp4} --vis_mode cam "
        f"--img_focal {intr['fx']} --static_camera",
        gpu, HAWOR_DIR, hawor_log,
        extra_env="export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 && ",
    )
    if not os.path.exists(npz):
        result["status"] = "failed"
        result["error"] = "HaWoR produced no all_hand_meshes.npz (see hawor.log)"
        return result
    # copy npz up to workdir root for a clean, discoverable location
    if not os.path.exists(npz_local):
        os.symlink(os.path.relpath(npz, wd), npz_local)

    masks = sorted(glob.glob(os.path.join(wd, "input", "tracks_*", "model_masks.npy")))
    mask_npy = masks[0] if masks else None

    # --- 2) smooth ---
    run_in_env("sam3d",
               f"python {SCRIPTS}/hand/smooth_hand_npz.py --input {npz} --output {smoothed}",
               gpu, wd, post_log)

    # --- 3) 3D MANO overlay (real intrinsics, smoothed) ---
    overlay = os.path.join(wd, "hand_mesh_overlay.mp4")
    run_in_env("sam3d",
               f"python {SCRIPTS}/render_hands_overlay.py --video {input_mp4} --npz {smoothed} "
               f"--output {overlay} --fx {intr['fx']} --fy {intr['fy']} "
               f"--cx {intr['cx']} --cy {intr['cy']}",
               gpu, wd, post_log)

    # --- 4) 2D hand mask overlay ---
    mask_video = os.path.join(wd, "hand_mask_overlay.mp4")
    if mask_npy:
        # raw mask already persisted at input/tracks_*/model_masks.npy — don't duplicate
        # the ~300MB array; the overlay video is the visual deliverable.
        run_in_env("sam3d",
                   f"python {SCRIPTS}/hand/export_hand_mask.py --video {input_mp4} "
                   f"--model-masks {mask_npy} --out-video {mask_video}",
                   gpu, wd, post_log)

    # --- 5) QC ---
    qc = os.path.join(wd, "qc_report.json")
    run_in_env("sam3d",
               f"python {SCRIPTS}/hand/hand_qc.py --smoothed {smoothed} --raw {npz} "
               f"--out {qc} --grid {os.path.join(wd, 'keyframes_grid.png')} "
               f"--overlay {overlay} --source {input_mp4}",
               gpu, wd, post_log)

    result["time_s"] = round(time.time() - t0, 1)
    result["status"] = "ok" if os.path.exists(qc) else "partial"
    if os.path.exists(qc):
        with open(qc) as f:
            result["qc"] = json.load(f)
    result["outputs"] = {
        "smoothed_npz": smoothed, "mesh_overlay": overlay,
        "mask_overlay": mask_video if mask_npy else None,
        "qc": qc,
    }
    return result


def worker(gpu, q, out_root, args, results):
    while True:
        try:
            clip = q.get_nowait()
        except queue.Empty:
            return
        log(f"[gpu{gpu}] START {clip.clip_id}")
        try:
            r = process_clip(clip, gpu, out_root, args)
        except Exception as e:
            r = {"clip": clip.clip_id, "gpu": gpu, "status": "error", "error": repr(e)}
        results.append(r)
        log(f"[gpu{gpu}] DONE  {clip.clip_id} -> {r.get('status')} ({r.get('time_s','?')}s)")
        q.task_done()


def write_summary(out_root, results):
    results = sorted(results, key=lambda r: r["clip"])
    with open(os.path.join(out_root, "SUMMARY.json"), "w") as f:
        json.dump(results, f, indent=2)
    lines = ["# 任务二 手部重建 — 批处理结果索引\n",
             f"共 {len(results)} 个 clip。每个目录: hand_mesh_overlay.mp4 / hand_mask_overlay.mp4 / "
             "all_hand_meshes_smoothed.npz / qc_report.json\n",
             "| clip | 状态 | 物体 | 锚手 | R valid | L valid | 抖动↓ | 异常帧 | 手性一致 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        qc = r.get("qc", {})
        rh = qc.get("hands", {}).get("right", {})
        lh = qc.get("hands", {}).get("left", {})
        lines.append(
            f"| {r['clip']} | {r.get('status')} | {r.get('object','-')} | {r.get('anchor_hand','-')} | "
            f"{rh.get('valid_frames','-')}/{rh.get('n_frames','-')} | "
            f"{lh.get('valid_frames','-')}/{lh.get('n_frames','-')} | "
            f"{rh.get('jitter_reduction_x','-')}x | {rh.get('n_anomaly_frames','-')} | "
            f"{qc.get('chirality_consistent','-')} |")
    with open(os.path.join(out_root, "INDEX.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    ap.add_argument("--sequences", default=None, help="comma list to restrict sequences")
    ap.add_argument("--cameras", default=None, help="comma list to restrict cameras")
    ap.add_argument("--limit", type=int, default=None, help="process only first N clips (smoke test)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    seqs = set(args.sequences.split(",")) if args.sequences else None
    cams = args.cameras.split(",") if args.cameras else None
    clips = enumerate_clips(args.data_root, cameras=cams, sequences=seqs)
    if args.limit:
        clips = clips[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    log(f"{len(clips)} clips on GPUs {gpus}")

    q = queue.Queue()
    for c in clips:
        q.put(c)
    results = []
    threads = [threading.Thread(target=worker, args=(g, q, args.out, args, results)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_summary(args.out, results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    log(f"\n=== DONE {ok}/{len(results)} ok. Summary: {os.path.join(args.out, 'SUMMARY.json')} ===")


if __name__ == "__main__":
    main()
