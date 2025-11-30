#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Language-Table (parquet + mp4)  →  TFDS 저장 스크립트

구조 가정:
  root/
    data/chunk-000/episode_XXXXX.parquet
    videos/chunk-000/observation.images.rgb/episode_XXXXX.mp4
"""

import argparse
from pathlib import Path
import os
import glob
import random

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import imageio.v2 as imageio  # AV1 / ffmpeg 기반 비디오 디코딩


def parse_args():
    p = argparse.ArgumentParser("Language-Table (parquet+mp4) → TFDS converter")

    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="hf download로 받은 language_table_lerobot의 루트 디렉토리",
    )
    p.add_argument(
        "--chunk-id",
        type=str,
        default="chunk-000",
        help="데이터/비디오 chunk 폴더 이름 (예: chunk-000)",
    )
    p.add_argument(
        "--tfds-root",
        type=Path,
        required=True,
        help="tf.data.Dataset.save() 결과를 저장할 최상위 디렉토리",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default="language_table",
        help="TFDS에서 쓸 dataset 이름 (예: language_table)",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="에피소드 샘플링 비율 (0 < fraction <= 1.0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="샘플링 랜덤 시드",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="이미지를 resize할 한 변 길이 (image_size x image_size)",
    )
    return p.parse_args()


def list_parquet_files(root: Path, chunk_id: str) -> list[Path]:
    data_dir = root / "data" / chunk_id
    files = sorted(data_dir.glob("episode_*.parquet"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No parquet episodes found under {data_dir}")
    return files


def infer_dims(sample_parquet: Path) -> tuple[int, int]:
    """parquet 하나 읽어서 state_dim, action_dim 추론"""
    df = pd.read_parquet(sample_parquet)
    if "observation.state" not in df.columns:
        raise KeyError(f"{sample_parquet}: 'observation.state' column not found")
    if "action" not in df.columns:
        raise KeyError(f"{sample_parquet}: 'action' column not found")

    state_dim = len(df["observation.state"].iloc[0])
    act_dim = len(df["action"].iloc[0])

    print(f"[INFO] inferred state_dim={state_dim}, act_dim={act_dim} from {sample_parquet.name}")
    return state_dim, act_dim


def read_video_frames(video_path: Path, image_size: int) -> np.ndarray:
    """
    imageio(ffmpeg)를 이용해서 AV1 mp4도 읽을 수 있게 한 함수.
    return: (T, H, W, 3) uint8
    """
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    try:
        reader = imageio.get_reader(str(video_path), format="ffmpeg")
    except Exception as e:
        raise IOError(f"cannot open video with imageio: {video_path}, err={e}")

    frames = []
    try:
        for frame in reader:
            # frame: (H, W, 3) RGB (imageio는 기본 RGB)
            if frame.ndim == 2:  # grayscale일 경우 3채널로 확장
                frame = np.stack([frame] * 3, axis=-1)

            frame_resized = cv2.resize(
                frame, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            frames.append(frame_resized.astype(np.uint8))
    finally:
        reader.close()

    if len(frames) == 0:
        raise ValueError(f"no frames decoded from {video_path}")

    return np.stack(frames, axis=0)  # (T, H, W, 3)


def load_episode(
    parquet_path: Path,
    video_path: Path,
    state_dim: int,
    act_dim: int,
    image_size: int = 128,
) -> dict:
    """
    parquet + 대응되는 mp4 하나를 읽어서
    CLAM이 좋아할 dict 형태로 변환
    """

    # ----- 1) parquet 읽기 -----
    df = pd.read_parquet(parquet_path)

    # state
    if "observation.state" in df.columns:
        states = np.stack(df["observation.state"].to_numpy(), axis=0).astype(np.float32)
        # 필요하다면 pad/truncate로 state_dim 맞추기
        if states.shape[1] < state_dim:
            pad = state_dim - states.shape[1]
            states = np.concatenate(
                [states, np.zeros((states.shape[0], pad), dtype=np.float32)],
                axis=1,
            )
        elif states.shape[1] > state_dim:
            states = states[:, :state_dim]
    else:
        # 없으면 그냥 zero state
        T_dummy = len(df)
        states = np.zeros((T_dummy, state_dim), dtype=np.float32)

    # action
    if "action" in df.columns:
        actions = np.stack(df["action"].to_numpy(), axis=0).astype(np.float32)
        if actions.shape[1] < act_dim:
            pad = act_dim - actions.shape[1]
            actions = np.concatenate(
                [actions, np.zeros((actions.shape[0], pad), dtype=np.float32)],
                axis=1,
            )
        elif actions.shape[1] > act_dim:
            actions = actions[:, :act_dim]
    else:
        actions = np.zeros((states.shape[0], act_dim), dtype=np.float32)

    T_p = states.shape[0]

    # ----- 2) 비디오 프레임 읽기 (imageio + ffmpeg 사용) -----
    frames = read_video_frames(video_path, image_size=image_size)  # (T_v, H, W, 3)
    T_v = frames.shape[0]

    # ----- 3) 길이 align -----
    T = min(T_p, T_v)
    if T <= 1:
        raise ValueError(f"T too small after sync: {T} (p={T_p}, v={T_v})")

    states = states[:T]
    actions = actions[:T]
    images = frames[:T]

    # ----- 4) reward / discount / flags -----
    rewards = np.zeros((T,), dtype=np.float32)
    discount = np.ones((T,), dtype=np.float32)

    is_first = np.zeros((T,), dtype=bool)
    is_last = np.zeros((T,), dtype=bool)
    is_terminal = np.zeros((T,), dtype=bool)

    is_first[0] = True
    is_last[-1] = True
    is_terminal[-1] = True

    ep = {
        "observations": states,     # (T, state_dim)
        "actions": actions,         # (T, act_dim)
        "rewards": rewards,         # (T,)
        "discount": discount,       # (T,)
        "is_first": is_first,       # (T,)
        "is_last": is_last,         # (T,)
        "is_terminal": is_terminal, # (T,)
        "images": images,           # (T, H, W, 3) uint8
    }
    return ep


def main():
    args = parse_args()

    root = args.root
    chunk_id = args.chunk_id

    parquet_files = list_parquet_files(root, chunk_id)
    print(f"[INFO] found {len(parquet_files)} parquet episodes under data/{chunk_id}")

    # fraction 적용
    if not (0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction must be in (0,1], got {args.fraction}")

    rnd = random.Random(args.seed)
    idxs_all = list(range(len(parquet_files)))
    rnd.shuffle(idxs_all)
    take_n = max(1, int(len(idxs_all) * args.fraction))
    idxs_sel = sorted(idxs_all[:take_n])
    sel_parquets = [parquet_files[i] for i in idxs_sel]

    print(f"[INFO] using {len(sel_parquets)} episodes (fraction={args.fraction})")

    # state_dim / act_dim 추론 (선택된 것 중 첫 번째)
    sample_parquet = sel_parquets[0]
    state_dim, act_dim = infer_dims(sample_parquet)

    video_dir = root / "videos" / chunk_id / "observation.images.rgb"

    # TFDS 저장 경로
    save_dir = args.tfds_root / args.dataset_name / chunk_id
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] TFDS will be saved to: {save_dir}")

    # output_signature 정의
    output_signature = {
        "observations": tf.TensorSpec(shape=(None, state_dim), dtype=tf.float32),
        "actions":      tf.TensorSpec(shape=(None, act_dim),   dtype=tf.float32),
        "rewards":      tf.TensorSpec(shape=(None,),           dtype=tf.float32),
        "discount":     tf.TensorSpec(shape=(None,),           dtype=tf.float32),
        "is_first":     tf.TensorSpec(shape=(None,),           dtype=tf.bool),
        "is_last":      tf.TensorSpec(shape=(None,),           dtype=tf.bool),
        "is_terminal":  tf.TensorSpec(shape=(None,),           dtype=tf.bool),
        "images":       tf.TensorSpec(
            shape=(None, args.image_size, args.image_size, 3), dtype=tf.uint8
        ),
    }

    def gen():
        skipped = 0
        for p in sel_parquets:
            ep_id = p.stem.split("_")[-1]   # episode_000123 → "000123"
            v_path = video_dir / f"episode_{ep_id}.mp4"

            try:
                ep = load_episode(
                    parquet_path=p,
                    video_path=v_path,
                    state_dim=state_dim,
                    act_dim=act_dim,
                    image_size=args.image_size,
                )
                yield ep
            except Exception as e:
                skipped += 1
                print(f"[WARN] skip episode {ep_id}: {e}")

        print(f"[STATS] total={len(sel_parquets)}, skipped={skipped}")

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # 최신 TF는 ds.save, 구버전은 experimental.save
    if hasattr(ds, "save"):
        ds.save(str(save_dir))
    else:
        tf.data.experimental.save(ds, str(save_dir))

    print(f"[DONE] TFDS saved at: {save_dir}")


if __name__ == "__main__":
    main()
