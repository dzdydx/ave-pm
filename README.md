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

1. **Clone the repository**

```bash
git clone git@github.com:dzdydx/ave-pm.git
cd ave-pm-main

conda env create -f environment.yaml
conda activate AVE-PM
```

## 📁 Dataset

You can download AVE-PM dataset from [Baidu Cloud Link](https://pan.baidu.com/s/1ErDp1zVEe0mugVMmQFbqow?pwd=2979). And then unzip the video files into the `dataset/data/videos/` folder.

### ⚙️ Data Preparation
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
python /LAVISH/scripts/extract_frames.py --out_dir /dataset/data/video_frames/ --video_dir dataset/data/videos/
python /LAVISH/scripts/extract_audio.py --video_pth dataset/data/videos/ --save_pth dataset/data/raw_audios
```

Output locations:
- Frames → `/dataset/data/video_frames/`
- Audios → `/dataset/data/raw_audios/`
  

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
python /LAVISH/scripts/extract_frames.py --out_dir dataset/data/AVE/video_frames --video_dir dataset/data/AVE/videos/
python /LAVISH/scripts/extract_audio.py --video_pth dataset/data/AVE/videos/ --save_pth dataset/data/AVE/raw_audios
```

Output locations:
- AVE Frames → `/dataset/data/AVE/video_frames/`
- AVE Audios → `/dataset/data/AVE/raw_audios/`

### 📂 Directory Structure

```graphql
AVE-PM/
├── dataset/
|	├── csvfiles/			   # csv files for training,validating and testing
|	|   ├── select/			  # csv files for S-LM and S-PM subsets
|	|   |  ├── ave/
|   |   |  └── avepm/
|   |	├── AVE/
|   |	|	├── data/          # AVE dataset csv files
|   |	|	├── feature/
|   |	|	├── raw_audios/
|   |	|	├── video_frames/
|   |	|	└── videos/        # AVE dataset videos
│   │   ├── videos/            # Raw portrait-mode videos
│   │   ├── video_frames/      # Extracted RGB frames (generated)
|	|	├── processed_audios/  # Preprocessed audio segments(generated)
│   │   └── raw_audios/        # Extracted audio segments (generated)
│   └── feature/			   # Precomputed audio and visual features
|       ├── features/  
|       ├── select/       # Precomputed features for S-LM and S-PM subsets
│       ├── preprocess_audio_feature/   
|       └── preprocess_visual_feature/
```

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
