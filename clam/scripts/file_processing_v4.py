# -*- coding: utf-8 -*-
"""
NPZ → TFDS converter (per-folder saving / streaming / 240x240 images)
- If root is a parent folder (e.g., eval/), convert every dataset subfolder inside it
  individually and save to <tfds-root>/<dataset-name>/<task-name>/<ds_dir.name>
- If root is a dataset folder (frames/ + index_mappings.json), convert only that folder
  and save to <tfds-root>/<dataset-name>/<task-name>/<root.name>
- If images exist, resize them to 240x240 and save as 'images' (uint8)
- If observations/actions are missing, create 39D/4D dummy arrays respectively
- On failure for a file, only print a warning and skip it
- To prevent OOM: use a generator without accumulating an episodes list →
  stream save via tf.data.Dataset.save
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import random

# ----------------- Args -----------------
def parse_args():
    p = argparse.ArgumentParser("Convert NPZ → TFDS (per-folder, streaming, 240x240 images)")
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="(1) A dataset folder that contains frames/ and index_mappings.json, or "
             "(2) A parent folder that contains multiple dataset folders (e.g., eval/)",
    )
    p.add_argument("--tfds-root", required=True, type=Path, help="TFDS root directory")
    p.add_argument("--dataset-name", required=True, type=str, help="Top-level TFDS dataset name")
    #p.add_argument("--task-name", required=True, type=str, help="Intermediate TFDS name (subfolder)")
    p.add_argument(
        "--task-filter",
        type=str,
        default=None,
        help='Only a specific task (e.g., "Open the door")',
    )
    p.add_argument(
        "--use-optimal",
        action="store_true",
        help="With task-filter, select only optimal trajectories",
    )
    p.add_argument(
        "--use-successful",
        action="store_true",
        help="Select only successful trajectories",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction to use from each dataset folder (0 < fraction <= 1). e.g., 0.2 = 20%",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed (xored with folder name for per-folder determinism)",
    )
    p.add_argument(
        "--no-shuffle",
        default=True,
        action="store_true",
        help="Do not shuffle before sampling (default: no shuffle)",
    )
    return p.parse_args()

# ----------------- Utils -----------------
def is_dataset_dir(d: Path) -> bool:
    return (d / "index_mappings.json").exists() and (d / "frames").exists()

def discover_dataset_dirs(root: Path):
    root = root.resolve()
    if is_dataset_dir(root):
        return [root]
    ds = [p for p in root.iterdir() if p.is_dir() and is_dataset_dir(p)]
    if not ds:
        raise FileNotFoundError(
            f"Could not find any dataset folders under {root}. "
            f"A dataset folder must contain both 'index_mappings.json' and 'frames/'."
        )
    return sorted(ds, key=lambda x: x.name.lower())

def build_filelist(root: Path):
    frames_dir = root / "frames"
    files = list(frames_dir.glob("trajectory_*.npz"))
    # Typo correction: ' . npz'
    for weird in frames_dir.glob("trajectory_*. npz"):
        fixed = Path(str(weird).replace(". npz", ".npz").strip())
        try:
            weird.rename(fixed)
        except Exception:
            pass
        files.append(fixed)
    files = sorted(set(files), key=lambda x: x.name.lower())
    if not files:
        raise FileNotFoundError(f"No trajectory_*.npz files found under {frames_dir}.")
    return files

def choose_indices(root: Path, args):
    meta = json.loads((root / "index_mappings.json").read_text())
    if args.task_filter:
        if args.use_optimal:
            pool = meta.get("optimal_by_task", {}).get(args.task_filter, [])
            if not pool:
                raise ValueError(f'No optimal indices for "{args.task_filter}" in {root.name}.')
            return pool
        pool = meta.get("task_indices", {}).get(args.task_filter, [])
        if not pool:
            raise ValueError(f'No task_indices for "{args.task_filter}" in {root.name}.')
        return pool
    if args.use_successful:
        pool = meta.get("quality_indices", {}).get("successful", [])
        if not pool:
            raise ValueError(f"No successful indices in {root.name}.")
        return pool
    pool = meta.get("robot_trajectories", [])
    if not pool:
        raise ValueError(f"robot_trajectories is empty in {root.name}.")
    return pool

def index_to_path(filelist, i: int) -> Path:
    if i < 0 or i >= len(filelist):
        raise IndexError(f"Index {i} out of range (0..{len(filelist)-1})")
    return filelist[i]

def _normalize_indices(filelist, idxs):
    if not idxs:
        return [], 0, False
    n = len(filelist)
    is_one_based = (0 not in idxs) and (max(idxs, default=-1) == n)
    idxs0 = [i - 1 for i in idxs] if is_one_based else list(idxs)
    filtered = [i for i in idxs0 if 0 <= i < n]
    dropped = len(idxs0) - len(filtered)
    return filtered, dropped, is_one_based

# ----------------- Core: NPZ -> episode dict -----------------
def npz_to_episode(p: Path):
    with np.load(p, allow_pickle=False) as d:
        ep = {}
        has_obs = "observations" in d.files
        has_img = "frames" in d.files
        has_act = "actions" in d.files
        has_rew = "rewards" in d.files

        T = -1

        # 1) images (240x240)
        if has_img:
            imgs = d["frames"]  # (T, H, W, C)
            target_hw = (128, 128)  # (W, H)
            resized = [cv2.resize(imgs[i], target_hw, interpolation=cv2.INTER_AREA)
                       for i in range(imgs.shape[0])]
            imgs_resized = np.asarray(resized, dtype=np.uint8)
            ep["images"] = imgs_resized  # (T,240,240,3)
            T = imgs_resized.shape[0]

        # 2) observations
        if has_obs:
            obs = d["observations"].astype(np.float32)
            ep["observations"] = obs
            if T == -1:
                T = obs.shape[0]
            else:
                assert obs.shape[0] == T, f"{p}: obs T {obs.shape[0]} != {T}"
        else:
            STATE_DIM = 39
            if T == -1:
                raise ValueError(
                    f"{p}: both 'frames' and 'observations' are missing, cannot determine length T."
                )
            ep["observations"] = np.zeros((T, STATE_DIM), dtype=np.float32)

        assert T > 0, f"{p}: invalid T"

        # 3) actions
        if has_act:
            acts = d["actions"].astype(np.float32)
            if acts.shape[0] != T:
                m = min(T, acts.shape[0])
                acts = acts[:m]
                for k in list(ep.keys()):
                    ep[k] = ep[k][:m]
                T = m
            ep["actions"] = acts
        else:
            ep["actions"] = np.zeros((T, 4), dtype=np.float32)  # 4D dummy

        # 4) rewards
        if has_rew:
            rew = d["rewards"].astype(np.float32)
            if rew.shape[0] != T:
                rew = rew[:T]
        else:
            rew = np.zeros((T,), dtype=np.float32)

        # 5) flags
        is_first = np.zeros((T,), dtype=bool); is_first[0] = True
        is_last  = np.zeros((T,), dtype=bool); is_last[-1] = True
        is_term  = np.zeros((T,), dtype=bool); is_term[-1] = True

        ep.update({
            "rewards": rew,
            "discount": np.ones((T,), dtype=np.float32),
            "is_first": is_first,
            "is_last":  is_last,
            "is_terminal": is_term,
        })
        return ep

# ----------------- Convert / save a single folder -----------------
def convert_one_folder(ds_dir: Path, args: argparse.Namespace):
    try:
        filelist = build_filelist(ds_dir)
        idxs_raw = choose_indices(ds_dir, args)
        idxs, dropped, was_one_based = _normalize_indices(filelist, idxs_raw)

        if not (0 < args.fraction <= 1.0):
            raise ValueError(f"--fraction must be in (0,1], got {args.fraction}")

        n_before = len(idxs)
        if n_before == 0:
            print(f"[INFO] [{ds_dir.name}] no selected indices.")
            return

        # Deterministic sampling per folder
        fold_seed = (hash(ds_dir.name) ^ args.seed) & 0xFFFFFFFF
        rnd = random.Random(fold_seed)

        idxs_local = list(idxs)
        if not args.no_shuffle:
            print("Shuffling ================================================================")
            rnd.shuffle(idxs_local)

        take_n = max(1, int(n_before * args.fraction)) if args.fraction < 1.0 else n_before
        idxs_sub = idxs_local[:take_n]

        print(
            f"[INFO] [{ds_dir.name}] files={len(filelist)}, selected(after_norm)={n_before}, "
            f"fraction={args.fraction} → using={len(idxs_sub)} (seed={fold_seed}, "
            f"{'no-shuffle' if args.no_shuffle else 'shuffled'})"
        )

        # Save path: <tfds-root>/<dataset-name>/<task-name>/<folder-name>
        save_dir = (args.tfds_root / args.dataset_name / ds_dir.name)
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Save path: {save_dir}")

        # Save signature
        output_signature = {
            "observations": tf.TensorSpec(shape=(None, 39), dtype=tf.float32),
            "actions":      tf.TensorSpec(shape=(None, 4),  dtype=tf.float32),
            "rewards":      tf.TensorSpec(shape=(None,),    dtype=tf.float32),
            "discount":     tf.TensorSpec(shape=(None,),    dtype=tf.float32),
            "is_first":     tf.TensorSpec(shape=(None,),    dtype=tf.bool),
            "is_last":      tf.TensorSpec(shape=(None,),    dtype=tf.bool),
            "is_terminal":  tf.TensorSpec(shape=(None,),    dtype=tf.bool),
            "images":       tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.uint8),
        }

        def gen_one_folder():
            skipped = 0
            for i in idxs_sub:
                try:
                    p = index_to_path(filelist, i)
                    yield npz_to_episode(p)
                except Exception as e:
                    skipped += 1
                    print(f"[WARN] Conversion failed for {ds_dir.name}[idx={i}]: {e}", file=sys.stderr)
            print(f"[STATS] [{ds_dir.name}] used={len(idxs_sub)}, skipped={skipped}")

        ds = tf.data.Dataset.from_generator(gen_one_folder, output_signature=output_signature)
        # Latest TF API
        if hasattr(tf.data.Dataset, "save"):
            ds.save(str(save_dir))
        else:
            tf.data.experimental.save(ds, str(save_dir))
        print(f"[DONE] [{ds_dir.name}] TFDS saved to: {save_dir}")

    except Exception as e:
        print(f"[WARN] Skipping folder {ds_dir}: {e}", file=sys.stderr)

# ----------------- Main -----------------
def main():
    args = parse_args()
    all_ds_dirs = discover_dataset_dirs(args.root)
    print(
        f"[INFO] Number of dataset folders to convert: {len(all_ds_dirs)} "
        f"@ {args.root.resolve()}"
    )

    for ds_dir in all_ds_dirs:
        convert_one_folder(ds_dir, args)

if __name__ == "__main__":
    main()
