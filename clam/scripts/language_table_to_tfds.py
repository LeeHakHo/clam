#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Language-Table Individual Chunk Converter (Fix: fraction argument added)
각 chunk-XXX를 별도의 TFDS 데이터셋으로 변환하여 저장합니다.
구조: {tfds_root}/{dataset_name}/samples/{chunk_id}
"""

import argparse
from pathlib import Path
import os
import random
import glob
import time

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import imageio.v2 as imageio
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser("Language-Table Chunk-by-Chunk Converter")

    p.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="language_table 루트 디렉토리 (data/ 와 videos/ 가 있는 곳)",
    )
    p.add_argument(
        "--tfds-root",
        type=Path,
        default=Path("tensorflow_datasets"),
        help="저장할 최상위 폴더 (예: tensorflow_datasets)",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default="language_table",
        help="데이터셋 이름 폴더",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="이미지 리사이즈 크기",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="데이터 사용 비율 (0.0 ~ 1.0). 기본값 1.0 (전체)",
    )
    return p.parse_args()


def get_sorted_chunk_ids(root: Path) -> list[str]:
    """data 폴더 내의 모든 chunk-XXX 폴더 이름을 정렬해서 반환"""
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist.")
    
    chunk_paths = sorted(data_dir.glob("chunk-*"))
    chunk_ids = [p.name for p in chunk_paths]
    return chunk_ids


def infer_dims(parquet_path: Path) -> tuple[int, int]:
    """첫 번째 파일로 차원 추론"""
    df = pd.read_parquet(parquet_path)
    state_dim = len(df["observation.state"].iloc[0]) if "observation.state" in df.columns else 1
    act_dim = len(df["action"].iloc[0]) if "action" in df.columns else 1
    return state_dim, act_dim


def read_video_frames(video_path: Path, image_size: int) -> np.ndarray:
    """비디오 디코딩"""
    try:
        reader = imageio.get_reader(str(video_path), format="ffmpeg")
        frames = []
        for frame in reader:
            frame_resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(frame_resized.astype(np.uint8))
        reader.close()
    except Exception:
        return np.zeros((0, image_size, image_size, 3), dtype=np.uint8)

    if not frames:
        return np.zeros((0, image_size, image_size, 3), dtype=np.uint8)
        
    return np.stack(frames, axis=0)


def load_episode(parquet_path: Path, video_path: Path, state_dim: int, act_dim: int, image_size: int) -> dict:
    """단일 에피소드 로드"""
    # 1. Parquet
    df = pd.read_parquet(parquet_path)
    if "observation.state" in df.columns:
        states = np.stack(df["observation.state"].to_numpy(), axis=0).astype(np.float32)
    else:
        states = np.zeros((len(df), state_dim), dtype=np.float32)

    if "action" in df.columns:
        actions = np.stack(df["action"].to_numpy(), axis=0).astype(np.float32)
    else:
        actions = np.zeros((len(df), act_dim), dtype=np.float32)

    # 2. Video
    frames = read_video_frames(video_path, image_size)

    # 3. Sync
    T = min(len(states), len(frames))
    if T < 1:
        raise ValueError("Empty episode")

    ep = {
        "observations": states[:T],
        "actions": actions[:T],
        "rewards": np.zeros((T,), dtype=np.float32),
        "discount": np.ones((T,), dtype=np.float32),
        "is_first": np.zeros((T,), dtype=bool),
        "is_last": np.zeros((T,), dtype=bool),
        "is_terminal": np.zeros((T,), dtype=bool),
        "images": frames[:T],
    }
    ep["is_first"][0] = True
    ep["is_last"][-1] = True
    ep["is_terminal"][-1] = True
    return ep


def process_single_chunk(root: Path, chunk_id: str, save_root: Path, state_dim: int, act_dim: int, image_size: int, fraction: float):
    """하나의 청크를 처리해서 TFDS로 저장"""
    
    # 경로 설정
    parquet_dir = root / "data" / chunk_id
    video_dir = root / "videos" / chunk_id / "observation.images.rgb"
    
    # 저장 경로: .../language_table/samples/chunk-000
    save_path = save_root / "samples" / chunk_id
    
    if save_path.exists():
        print(f"[SKIP] {chunk_id} already exists at {save_path}")
        return

    # 파일 리스트업
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    
    # Fraction 적용 (랜덤 샘플링)
    if fraction < 1.0:
        random.shuffle(parquet_files)
        take_n = max(1, int(len(parquet_files) * fraction))
        parquet_files = parquet_files[:take_n]

    if not parquet_files:
        print(f"[WARN] No files in {chunk_id}, skipping.")
        return

    # Signature 정의
    output_signature = {
        "observations": tf.TensorSpec(shape=(None, state_dim), dtype=tf.float32),
        "actions":      tf.TensorSpec(shape=(None, act_dim), dtype=tf.float32),
        "rewards":      tf.TensorSpec(shape=(None,), dtype=tf.float32),
        "discount":     tf.TensorSpec(shape=(None,), dtype=tf.float32),
        "is_first":     tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "is_last":      tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "is_terminal":  tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "images":       tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.uint8),
    }

    # Generator 생성
    def generator():
        for p_path in parquet_files:
            ep_id = p_path.stem.split("_")[-1]
            v_path = video_dir / f"episode_{ep_id}.mp4"
            
            try:
                # 비디오가 없거나 깨졌으면 건너뜀
                if not v_path.exists(): continue
                
                yield load_episode(p_path, v_path, state_dim, act_dim, image_size)
            except Exception:
                continue

    # 데이터셋 생성 및 저장
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    # TFDS 저장 (폴더 생성 포함)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save(str(save_path))


def main():
    args = parse_args()
    root = args.root.resolve()
    tfds_root = args.tfds_root / args.dataset_name # e.g. tensorflow_datasets/language_table

    print(f"=== Starting Batch Conversion ===")
    print(f"Source: {root}")
    print(f"Target: {tfds_root}/samples/chunk-XXX")
    print(f"Fraction: {args.fraction}")

    # 1. Chunk ID 확인
    try:
        chunk_ids = get_sorted_chunk_ids(root)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Please run this script from the 'language_table' directory.")
        return

    print(f"Found {len(chunk_ids)} chunks: {chunk_ids[0]} ~ {chunk_ids[-1]}")

    # 2. 차원 추론 (첫 번째 청크의 첫 파일 사용)
    first_parquet = next((root / "data" / chunk_ids[0]).glob("episode_*.parquet"))
    state_dim, act_dim = infer_dims(first_parquet)
    print(f"Dimensions inferred: State={state_dim}, Action={act_dim}")

    # 3. 각 청크별 처리 (Progress Bar)
    pbar = tqdm(chunk_ids)
    for chunk_id in pbar:
        pbar.set_description(f"Processing {chunk_id}")
        
        process_single_chunk(
            root=root,
            chunk_id=chunk_id,
            save_root=tfds_root,
            state_dim=state_dim,
            act_dim=act_dim,
            image_size=args.image_size,
            fraction=args.fraction  # 추가된 부분
        )

    print("\n[DONE] All chunks processed.")
    print(f"Check results in: {tfds_root}/samples/")

if __name__ == "__main__":
    main()