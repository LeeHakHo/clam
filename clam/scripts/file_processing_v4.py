# -*- coding: utf-8 -*-
"""
NPZ → TFDS 변환기 (폴더별 개별 저장 / 스트리밍 / 240x240 이미지)
- root가 상위 폴더(e.g., eval/)면 내부의 모든 데이터셋 폴더를 '각각' 변환하여
  <tfds-root>/<dataset-name>/<task-name>/<ds_dir.name> 에 저장
- root가 데이터셋 폴더(frames/ + index_mappings.json)면 그 폴더만 변환하여
  <tfds-root>/<dataset-name>/<task-name>/<root.name> 에 저장
- 이미지가 있으면 240x240로 리사이즈하여 'images'로 저장 (uint8)
- observations/actions 없으면 각각 39D/4D 더미 생성
- 실패 파일은 경고만 내고 건너뜀
- OOM 방지: episodes 리스트 누적 없이 제너레이터 → tf.data.Dataset.save 스트리밍 저장
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import random

# ----------------- 인자 -----------------
def parse_args():
    p = argparse.ArgumentParser("Convert NPZ → TFDS (per-folder, streaming, 240x240 images)")
    p.add_argument("--root", required=True, type=Path,
                   help="(1) frames/와 index_mappings.json이 있는 데이터셋 폴더 또는 "
                        "(2) 여러 데이터셋 폴더를 포함한 상위 폴더(예: eval/)")
    p.add_argument("--tfds-root", required=True, type=Path, help="TFDS 루트 디렉터리")
    p.add_argument("--dataset-name", required=True, type=str, help="TFDS 상위 이름")
    #p.add_argument("--task-name", required=True, type=str, help="TFDS 중간 이름(서브폴더)")
    p.add_argument("--task-filter", type=str, default=None, help='특정 태스크만 (예: "Open the door")')
    p.add_argument("--use-optimal", action="store_true", help="task-filter와 함께 optimal만 선택")
    p.add_argument("--use-successful", action="store_true", help="성공한 궤적만 선택")
    p.add_argument("--fraction", type=float, default=0.1,
                   help="각 데이터셋 폴더에서 사용할 비율 (0<fraction<=1). 예: 0.2 = 20%")
    p.add_argument("--seed", type=int, default=42, help="샘플링 시드(폴더명과 xor하여 폴더별 고정)")
    p.add_argument("--no-shuffle", default=True, action="store_true",
                   help="샘플링 전에 셔플하지 않음(기본은 셔플x)")
    return p.parse_args()

# ----------------- 유틸 -----------------
def is_dataset_dir(d: Path) -> bool:
    return (d / "index_mappings.json").exists() and (d / "frames").exists()

def discover_dataset_dirs(root: Path):
    root = root.resolve()
    if is_dataset_dir(root):
        return [root]
    ds = [p for p in root.iterdir() if p.is_dir() and is_dataset_dir(p)]
    if not ds:
        raise FileNotFoundError(
            f"{root} 아래에서 데이터셋 폴더를 찾지 못했습니다. "
            f"데이터셋 폴더는 'index_mappings.json'과 'frames/'를 포함해야 합니다."
        )
    return sorted(ds, key=lambda x: x.name.lower())

def build_filelist(root: Path):
    frames_dir = root / "frames"
    files = list(frames_dir.glob("trajectory_*.npz"))
    # 오타 교정: ' . npz'
    for weird in frames_dir.glob("trajectory_*. npz"):
        fixed = Path(str(weird).replace(". npz", ".npz").strip())
        try:
            weird.rename(fixed)
        except Exception:
            pass
        files.append(fixed)
    files = sorted(set(files), key=lambda x: x.name.lower())
    if not files:
        raise FileNotFoundError(f"{frames_dir} 아래에 trajectory_*.npz가 없습니다.")
    return files

def choose_indices(root: Path, args):
    meta = json.loads((root / "index_mappings.json").read_text())
    if args.task_filter:
        if args.use_optimal:
            pool = meta.get("optimal_by_task", {}).get(args.task_filter, [])
            if not pool:
                raise ValueError(f'{root.name}에서 "{args.task_filter}"의 optimal 인덱스가 없습니다.')
            return pool
        pool = meta.get("task_indices", {}).get(args.task_filter, [])
        if not pool:
            raise ValueError(f'{root.name}에서 "{args.task_filter}"의 task_indices가 없습니다.')
        return pool
    if args.use_successful:
        pool = meta.get("quality_indices", {}).get("successful", [])
        if not pool:
            raise ValueError(f"{root.name}에 successful 인덱스가 없습니다.")
        return pool
    pool = meta.get("robot_trajectories", [])
    if not pool:
        raise ValueError(f"{root.name}의 robot_trajectories가 비어있습니다.")
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

# ----------------- 핵심: NPZ -> episode dict -----------------
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
            if T == -1: T = obs.shape[0]
            else: assert obs.shape[0] == T, f"{p}: obs T {obs.shape[0]} != {T}"
        else:
            STATE_DIM = 39
            if T == -1:
                raise ValueError(f"{p}: 'frames'와 'observations'가 모두 없어 길이 T를 알 수 없습니다.")
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
            ep["actions"] = np.zeros((T, 4), dtype=np.float32)  # 4D 더미

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

# ----------------- 폴더 하나를 개별 변환/저장 -----------------
def convert_one_folder(ds_dir: Path, args: argparse.Namespace):
    try:
        filelist = build_filelist(ds_dir)
        idxs_raw = choose_indices(ds_dir, args)
        idxs, dropped, was_one_based = _normalize_indices(filelist, idxs_raw)

        if not (0 < args.fraction <= 1.0):
            raise ValueError(f"--fraction must be in (0,1], got {args.fraction}")

        n_before = len(idxs)
        if n_before == 0:
            print(f"[INFO] [{ds_dir.name}] 선택된 인덱스가 없습니다."); 
            return

        # 폴더별 결정적 샘플링
        fold_seed = (hash(ds_dir.name) ^ args.seed) & 0xFFFFFFFF
        rnd = random.Random(fold_seed)

        idxs_local = list(idxs)
        if not args.no_shuffle:
            print("셔플함====================================================================")
            rnd.shuffle(idxs_local)

        take_n = max(1, int(n_before * args.fraction)) if args.fraction < 1.0 else n_before
        idxs_sub = idxs_local[:take_n]

        print(
            f"[INFO] [{ds_dir.name}] files={len(filelist)}, selected(after_norm)={n_before}, "
            f"fraction={args.fraction} → using={len(idxs_sub)} (seed={fold_seed}, "
            f"{'no-shuffle' if args.no_shuffle else 'shuffled'})"
        )

        # 저장 경로: <tfds-root>/<dataset-name>/<task-name>/<폴더명>
        save_dir = (args.tfds_root / args.dataset_name / ds_dir.name)
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 저장 경로: {save_dir}")

        # 저장 시그니처
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
                    print(f"[WARN] 변환 실패 {ds_dir.name}[idx={i}]: {e}", file=sys.stderr)
            print(f"[STATS] [{ds_dir.name}] used={len(idxs_sub)}, skipped={skipped}")

        ds = tf.data.Dataset.from_generator(gen_one_folder, output_signature=output_signature)
        # TF 최신 API
        if hasattr(tf.data.Dataset, "save"):
            ds.save(str(save_dir))
        else:
            tf.data.experimental.save(ds, str(save_dir))
        print(f"[DONE] [{ds_dir.name}] TFDS saved to: {save_dir}")

    except Exception as e:
        print(f"[WARN] 폴더 스킵 {ds_dir}: {e}", file=sys.stderr)

# ----------------- 메인 -----------------
def main():
    args = parse_args()
    all_ds_dirs = discover_dataset_dirs(args.root)
    print(f"[INFO] 변환 대상 데이터셋 폴더: {len(all_ds_dirs)}개 @ {args.root.resolve()}")

    for ds_dir in all_ds_dirs:
        convert_one_folder(ds_dir, args)

if __name__ == "__main__":
    main()
