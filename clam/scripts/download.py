from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="IPEC-COMMUNITY/language_table_lerobot",
#     repo_type="dataset",
#     local_dir="/project2/biyik_1165/hyeonhoo/language_table_30",
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     allow_patterns="videos/*" # [중요] data 폴더 제외하고 videos 폴더만 다운로드
# )

# 1. language_table 폴더로 이동
cd /project2/biyik_1165/hyeonhoo/language_table_30

# 2. CLI로 누락된 파일 다시 받기 (로그인 상태여야 함)
huggingface-cli download IPEC-COMMUNITY/language_table_lerobot \
    --repo-type dataset \
    --local-dir . \
    --local-dir-use-symlinks False \
    --resume-download