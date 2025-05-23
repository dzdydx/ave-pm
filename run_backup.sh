
# !/bin/bash


# select baselines and run experiments
BASELINE="" # [AVEL, CMBS, LAVISH, CPSP]

# set environment variables
SEED=42
SANPSHOT_PREF=""

# DATA PATH
DATA_ROOT="dataset/feature/features/"
META_ROOT="dataset/csvfiles/"
AVE=False
AVEPM=True
IS_SELECT=False
AUDIO_PREPROCESS_MODE="None"
PREPROCESS="None"


# cpsp/AVEL/CMBS
V_FEATURE_ROOT=""
A_FEATURE_ROOT=""

# LAVISH
AUDIO_FOLDER=""
VIDEO_FOLDER=""
PROCESSED_AUDIO_ROOT=""

if [ "$BASELINE" = "CPSP" ]; then
    cd ./CPSP
    # overide config parameters

    # then 
    python main.py --config config/cpsp.yaml
