# AVE-PM: An Audio-visual Event Localization Dataset for Portrait Mode Short Videos

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

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
Before training and testing, you need to preprocess the data by extracting audio and visual features.

🔉 **Feature Extraction**

Initially, you can use the provided feature extraction script to extract features from the raw audio and visual features in AVEPM dataset. The script is located at `scripts_helper/encode.py`

``` bash 
bash scripts_helper/get_raw_feature.sh
```

This script will extract audio and visual features from the raw videos and store them in the `dataset/data/features` folder.

You can also use your own feature extraction method if desired.

🧠 **Event Template and Preprocessing**
Secondly, you need to get the `event_templates.pkl` file for audio preprocessing. And then, you can process audio/video and extract audio/visual features in different methods using the following command:
``` bash
# Step 1: get the event_templates.pkl file
bash scripts_helper/get_template.sh

# Step 2: process audio/video and extract audio/visual features
bash scripts_helper/run_preprocess.sh
```
You may modify the parameters in the `.sh` files to suit your needs.

You may also use your own feature extraction method if desired.


🎞️ **Extract Frames & Audio for LAVISH**

Before training the LAVISH model, you should extract video frames and raw audios, and putting them into `/dataset/data/video_frames` and `/dataset/data/raw_audios` folder.

```python
python /LAVISH/scripts/extract_frames.py --out_dir /dataset/data/video_frames/ --video_dir dataset/data/videos/
python /LAVISH/scripts/extract_audio.py --video_pth dataset/data/videos/ --save_pth dataset/data/raw_audios
```
The outputs will be saved to:

- Frames → `/dataset/data/video_frames/`
- Audios → `/dataset/data/raw_audios/`
  

### ⚙️ Cross-mode evaluation
To demonstrate the domain differences between landscape mode (LM) and portrait mode(PM) videos in the context of audio-visual event localization, we conducted a cross-mode evaluation on the S-LM and S-PM subsets. For a rigorous comparison, we selected 10 overlapping categories from AVE dataset and AVE-PM. Initially, we ultilize all samples from the corresponding categories of the AVE dataset to build the S-LM, which comprises 1536 samples, accounting for 37% of the total 4143 samples in AVE dataset. Subsequently, we select an equal number of samples per category from the AVE-PM dataset to build the S-PM.

Initially, you need to download the AVE dataset and put videos under `dataset/data/AVE/videos` folder. Then, you can use the following command to generate features for S-LM and S-PM subsets:

```bash 
bash scripts_helper/get_select_dataset.sh
```
This will generate:
- AVE Features → `dataset/data/AVE/feature`
- S-LM Features → `dataset/feature/select/ave`
- S-PM Features → `dataset/feature/select/avepm`

Extract raw frames and raw audios from AVE dataset:
```python
python /LAVISH/scripts/extract_frames.py --out_dir dataset/data/AVE/video_frames --video_dir dataset/data/AVE/videos/
python /LAVISH/scripts/extract_audio.py --video_pth dataset/data/AVE/videos/ --save_pth dataset/data/AVE/raw_audios
```
The outputs will be saved to:

- AVE Frames → `/dataset/data/AVE/video_frames/`
- AVE Audios → `/dataset/data/AVE/raw_audios/`

### 📂 Directory Layout

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

We provide training and inference scripts for four baseline models evaluated on AVE-PM. You can use the following command to train or evaluate the models:

```bash
bash run.sh # Customize model and mode inside the script
```
Edit run.sh to set the model name and training/evaluation options.

**Note**: The `run.sh` script serves as a wrapper to configure the baseline model and specify whether to train or evaluate. You can customize its behavior by modifying the corresponding parameters in the corresponding scripts.

### ✅ Pretrained Checkpoints

You can download pretrained model checkpoints here:
 | Model | Checkpoint | 
 | --- | --- | 
 | AVEL | *(Coming Soon)* | 
 | CPSP | *(Coming Soon)* | 
 | CMBS | *(Coming Soon)* | 
 | LAVISH | *(Coming Soon)* | 



## 📌 Citation

If you find this project helpful for your research, please consider citing:

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

We thank the authors of the following methods and datasets for their valuable contributions:

- **AVEL (ECCV'18):**
   https://github.com/YapengTian/AVE-ECCV18
- **CPSP (CPSP@CVPR'22):**
   https://github.com/jasongief/cpsp
- **CMBS (CMBS@ICCV'23):**
   https://github.com/marmot-xy/CMBS
- **LAVISH (LAVISH@ICCV'23):**
   https://genjib.github.io/project_page/LAVISH/
- **ByteDance Portrait-Mode Video Recognition:**
   https://github.com/bytedance/portrait-video-recognition



