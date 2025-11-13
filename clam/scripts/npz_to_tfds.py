import argparse, json
from pathlib import Path
import numpy as np
from clam.scripts.data import save_dataset
import cv2

def parse_args():
    p = argparse.ArgumentParser("Convert NPZ → TFDS")
    p.add_argument("--root", required=True, type=Path, help="NPZ가 있는 폴더 (train/eval)")
    p.add_argument("--tfds-root", required=True, type=Path, help="TFDS 루트")
    p.add_argument("--dataset-name", required=True, type=str, help="TFDS 상위 이름")
    p.add_argument("--task-name", required=True, type=str, help="TFDS 하위 이름")
    p.add_argument("--task-filter", type=str, default=None, help='특정 태스크만 (예: "Open the door")')
    p.add_argument("--use-optimal", action="store_true", help="task-filter와 함께 optimal만")
    p.add_argument("--use-successful", action="store_true", help="성공한 궤적만")
    p.add_argument("--images-as-observations", action="store_true",
                help="observations가 없으면 images/255.0을 관측으로 사용")
    return p.parse_args()

def build_filelist(root: Path):
    files = list(root.glob("frames/trajectory_*.npz"))
    # 간혹 ' . npz' 같은 오타 처리
    for weird in root.glob("trajectory_*. npz"):
        fixed = Path(str(weird).replace(". npz", ".npz").strip())
        try: weird.rename(fixed)
        except Exception: pass
        files.append(fixed)
    files = sorted(set(files), key=lambda x: x.name.lower())
    if not files:
        raise FileNotFoundError(f"No trajectory_*.npz under {root}")
    return files

def choose_indices(root: Path, args):
    meta = json.loads((root/"index_mappings.json").read_text())
    if args.task_filter:
        if args.use_optimal:
            pool = meta.get("optimal_by_task", {}).get(args.task_filter, [])
            if not pool: raise ValueError(f'No optimal for "{args.task_filter}"')
            return pool
        pool = meta.get("task_indices", {}).get(args.task_filter, [])
        if not pool: raise ValueError(f'No task_indices for "{args.task_filter}"')
        return pool
    if args.use_successful:
        pool = meta.get("quality_indices", {}).get("successful", [])
        if not pool: raise ValueError("No successful indices")
        return pool
    pool = meta.get("robot_trajectories", [])
    if not pool: raise ValueError("robot_trajectories empty")
    return pool

def index_to_path(filelist, i: int) -> Path:
    if i < 0 or i >= len(filelist):
        raise IndexError(f"Index {i} out of range (0..{len(filelist)-1})")
    return filelist[i]

#배열 이름: 'frames'
#- 형태(Shape): (32, 240, 240, 3)
#- 타입(Dtype): uint8

# [전체 교체]
# scripts/npz_to_tfds.py 파일의 npz_to_episode 함수를
# 아래 내용으로 완전히 덮어쓰세요.

def npz_to_episode(p: Path):
    """
    --images-as-observations 플래그 인자를 제거하고,
    observations와 actions가 없을 경우 더미 데이터를 생성하도록 수정
    """
    with np.load(p, allow_pickle=False) as d:
        ep = {}
        has_obs = "observations" in d.files
        has_img = "frames" in d.files
        has_act = "actions" in d.files
        has_rew = "rewards" in d.files

        # --- 1. T (길이) 결정 및 이미지 리사이즈 (84x84) ---
        T = -1
        if has_img:
            imgs = d["frames"]       # (T, 240, 240, 3) uint8
            
            TARGET_SIZE = (84, 84) # (width, height)
            resized_imgs_list = []
            
            for i in range(imgs.shape[0]):
                resized_frame = cv2.resize(imgs[i], TARGET_SIZE, interpolation=cv2.INTER_AREA)
                resized_imgs_list.append(resized_frame)
            
            imgs_resized_np = np.array(resized_imgs_list, dtype=np.uint8)

            ep["images"] = imgs_resized_np  # (T, 84, 84, 3)
            T = imgs_resized_np.shape[0]
        
        # --- 2. 'observations' 처리 (없으면 39차원 더미 생성) ---
        if has_obs:
            obs = d["observations"].astype(np.float32)
            ep["observations"] = obs
            if T == -1: # 이미지가 없었을 경우
                T = obs.shape[0]
        else:
            # 에러 로그에서 (None, 39)를 기대했음
            STATE_DIM = 39 
            if T == -1:
                raise ValueError(f"NPZ 파일 {p}에 'frames'와 'observations'가 모두 없어 길이를 알 수 없습니다.")
            
            # (T, 39) 크기의 0으로 채워진 배열 생성
            ep["observations"] = np.zeros((T, STATE_DIM), dtype=np.float32)

        if T == -1:
            raise ValueError(f"NPZ 파일 {p}에서 'frames'나 'observations'를 찾을 수 없어 길이를 알 수 없습니다.")

        # --- 3. 'actions' 처리 (없으면 4차원 더미 생성) ---
        if has_act:
            ep["actions"] = d["actions"].astype(np.float32)
        else:
            # 참조 스크립트에서 Metaworld action이 4D임을 확인
            ACTION_DIM = 4 
            ep["actions"] = np.zeros((T, ACTION_DIM), dtype=np.float32)

        # --- 4. 'rewards' 처리 (기존 코드) ---
        if has_rew:
            rew = d["rewards"].astype(np.float32)
        else:
            rew = np.zeros((T,), dtype=np.float32)
        
        # (이후 코드는 동일)
        ep.update({
            "rewards": rew,
            "discount": np.ones((T,), dtype=np.float32),
            "is_first": np.zeros((T,), dtype=np.int32),
            "is_last":  np.zeros((T,), dtype=np.int32),
            "is_terminal": np.zeros((T,), dtype=np.int32),
        })
        ep["is_first"][0] = 1; ep["is_last"][-1] = 1; ep["is_terminal"][-1] = 1
        return ep

def main():
    args = parse_args()
    root = args.root.resolve()
    filelist = build_filelist(root)
    idxs = choose_indices(root, args)
    print(f"[INFO] root={root}")
    print(f"[INFO] files={len(filelist)}, episodes(selected)={len(idxs)}")

    paths = [index_to_path(filelist, i) for i in idxs]
    episodes = [npz_to_episode(p) for p in paths]

    save_dir = (args.tfds_root / args.dataset_name / args.task_name)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    save_imgs = any("images" in ep for ep in episodes)
    save_dataset(episodes, save_dir, env_name="metaworld",
            save_imgs=save_imgs, framestack=1)
    print(f"[DONE] TFDS saved to: {save_dir}")

if __name__ == "__main__":
    main()