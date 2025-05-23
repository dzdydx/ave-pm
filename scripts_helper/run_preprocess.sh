#!/bin/bash
set -e  # 一旦出错就停止运行

# -----------------------------
# Step 1: Audio Preprocessing
# -----------------------------
python scripts_helper/audio_preprocess.py \
    --annotation_file "dataset/csvfiles/annotations.csv" \
    --preprocess_type "NMF1" \
    --template_path "dataset/event_templates.pkl" \
    --output_root "dataset/data/processed_audios/" \
    --videos_root_dir "dataset/data/videos/" \
    --n_fft 2048 \
    --hop_length 512 \
    --LMS_filter_length 128 \
    --LMS_step_size 0.01 \
    --LMS_iterations 1000 \
    --threshold1 0.3 \
    --threshold2 0.1 \
    --threshold3 0.7 \
    --feature_root "dataset/feature/preprocess_audio_feature/"

# -----------------------------
# Step 2: Visual Preprocessing
# -----------------------------
SEED=42
python scripts_helper/visual_preprocess.py \
    --csv_path "dataset/csvfiles/meta_data_selected.csv" \
    --frame_dir "dataset/data/selected_pm_frames/" \
    --video_root "dataset/data/videos/" \
    --feature_root "dataset/feature/preprocess_visual_feature/" \
    --random_seed $SEED
