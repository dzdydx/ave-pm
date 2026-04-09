# AVE-PM: An Audio-visual Event Localization Dataset for Portrait Mode Short Videos

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-green.svg)

**Update**: The AVE-PM project page is now available - [check it out!](https://dzdydx.github.io/ave-pm-homepage/)

**AVE-PM** is the first audio-visual event localization (AVEL) dataset specifically designed for **portrait-mode short videos**. It contains:

- 📹 **25,335** video clips
- 🔊 **86** fine-grained event categories
- ⏱️ **Frame-level annotations** for precise temporal localization
- 🎶 **Sample-level annotations** indicating background music presence

---

## 📦 Installation

```bash
git clone git@github.com:dzdydx/ave-pm.git
cd ave-pm

conda env create -f environment.yaml
conda activate AVE-PM
```

## 📁 Dataset

The AVE-PM dataset is built upon the [Portrait-Mode 400 (PM-400)](https://github.com/bytedance/Portrait-Mode-Video) dataset, with videos sourced from the Douyin platform. Directly distributing the raw videos may violate Douyin's terms of use, so we provide download scripts and video IDs following the same approach as PM-400.

### ⚙️ Prepare Data

**Step 1: Download PM-400 videos from source**

There are two options to download the original PM-400 videos:
1. (Recommended) Follow the instructions below. Click-to-run scripts are provided. Only necessary videos (~20% of the whole dataset) are downloaded here.
2. Alternatively, a community-uploaded cache of PM-400 dataset can be accessed [here](https://github.com/bytedance/Portrait-Mode-Video/issues/7).

Either way, please place the downloaded videos at `dataset/PM-400/videos` to proceed to following steps without modifying default paths.

We provide video links in `dataset/PM-400/video_links_filtered.csv`. Use the provided script to download the videos:

```bash
python scripts_helper/download_videos.py \
    --video_links dataset/PM-400/video_links_filtered.csv \
    --output_dir dataset/PM-400/videos/
```

Failed downloads will be logged in `dataset/PM-400/fail_cases.csv` for retry attempts.

> [!NOTE] 
> Network issues may happen when downloading original PM-400 videos. Thanks to the community, a cached version of AVE-PM is provided. Check [the issue here](https://github.com/dzdydx/ave-pm/issues/1) to download the processed clips of AVE-PM. You can skip step 2 if you choose this way.

**Step 2: Prepare AVE-PM clips**

Cut 10-second video clips from the original PM-400 videos using the provided annotations. Each clip is named by its `sample_id`:

```bash
python scripts_helper/cut_video_files.py --cut_clips \
    --videos_dir dataset/PM-400/videos \
    --annotation_csv dataset/PM-400/PM400-to-AVEPM.csv \
    --out_videos_dir dataset/AVE-PM/videos
```

This reads the per-clip annotations (start/end times, onset/offset) and cuts the corresponding segments from the source videos. The script resumes automatically if interrupted.

**Step 3: Extract features and preprocess**

Prior to training and evaluation, you'll need to preprocess the data by extracting audio and visual features.

🔉 **Feature Extraction**

Use the provided script to extract features from raw audio and video data. The script is located at `scripts_helper/encode.py`:

``` bash 
bash scripts_helper/get_raw_feature.sh
```

This will extract audio and visual features from the videos and store them in `dataset/data/features`.

🧠 **Event Template and Preprocessing**
First generate the `event_templates.pkl` file for audio preprocessing, then process the audio/video data:

``` bash
# Step 1: get the event_templates.pkl file
bash scripts_helper/get_template.sh

# Step 2: process audio/video and extract audio/visual features
bash scripts_helper/run_preprocess.sh
```

Customize the parameters in the `.sh` files as needed. Alternative feature extraction methods may be used.


🎞️ **Video Frames & Audio Extraction for LAVISH**

For LAVISH model training, extract video frames and raw audio into respective directories:

```python
python LAVISH/scripts/extract_frames.py --out_dir dataset/AVE-PM/video_frames/ --video_dir dataset/AVE-PM/videos/
python LAVISH/scripts/extract_audio.py --video_pth dataset/AVE-PM/videos/ --save_pth dataset/AVE-PM/raw_audios
```

Output locations:
- Frames → `dataset/AVE-PM/video_frames/`
- Audios → `dataset/AVE-PM/raw_audios/`
  

### ⚙️ Cross-mode evaluation
To investigate domain differences between landscape (LM) and portrait mode (PM) videos in audio-visual event localization, we conducted cross-mode evaluations using the S-LM and S-PM subsets. We selected 10 overlapping categories from both AVE and AVE-PM datasets, creating:

1. S-LM: All samples from AVE's corresponding categories (1,536 samples, 37% of total AVE dataset)
2. S-PM: Balanced samples from AVE-PM's matching categories

First download the AVE dataset to `dataset/data/AVE/videos`, then generate features:

```bash 
bash scripts_helper/get_select_dataset.sh
```

This creates:
- AVE Features → `dataset/data/AVE/feature`
- S-LM Features → `dataset/feature/select/ave`
- S-PM Features → `dataset/feature/select/avepm`

Extract AVE dataset frames and audio:

```python
python LAVISH/scripts/extract_frames.py --out_dir dataset/data/AVE/video_frames --video_dir dataset/data/AVE/videos/
python LAVISH/scripts/extract_audio.py --video_pth dataset/data/AVE/videos/ --save_pth dataset/data/AVE/raw_audios
```

Output locations:
- AVE Frames → `dataset/data/AVE/video_frames/`
- AVE Audios → `dataset/data/AVE/raw_audios/`

## 🚀 Training & Evaluation

We provide scripts for four baseline models. Run evaluations using:

```bash
bash run.sh # Customize model and mode inside the script
```

Modify `run.sh` to select models and specify training/evaluation modes.

**Note**: The `run.sh` script serves as a configuration wrapper. Adjust parameters in the corresponding scripts as needed.

### ✅ Pretrained Checkpoints

Available checkpoints:
 | Model | Checkpoint | 
 | --- | --- | 
 | AVEL | *(Coming Soon)* | 
 | CPSP | *(Coming Soon)* | 
 | CMBS | *(Coming Soon)* | 
 | LAVISH | *(Coming Soon)* | 



## 📌 Citation

If you find this project helpful, please consider citing:

```bibtex
@misc{liu2025audiovisualeventlocalizationportrait,
      title={Audio-visual Event Localization on Portrait Mode Short Videos}, 
      author={Wuyang Liu and Yi Chai and Yongpeng Yan and Yanzhen Ren},
      year={2025},
      eprint={2504.06884},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2504.06884}, 
}
```



## 🙏 Acknowledgements

We acknowledge the following foundational works:

- **AVEL (ECCV'18):**
   https://github.com/YapengTian/AVE-ECCV18
- **CPSP (CPSP@CVPR'22):**
   https://github.com/jasongief/cpsp
- **CMBS (CMBS@ICCV'23):**
   https://github.com/marmot-xy/CMBS
- **LAVISH (LAVISH@ICCV'23):**
   https://genjib.github.io/project_page/LAVISH/
- **ByteDance Portrait-Mode Video Recognition:**
   https://github.com/bytedance/Portrait-Mode-Video


## 📄 License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).
