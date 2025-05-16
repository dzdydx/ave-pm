# AVE-PM: An Audio-visual Event Localization Dataset for Portrait Mode Short Videos

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**AVE-PM** is the first audio-visual event localization (AVEL) dataset specifically designed for **portrait-mode short videos**. It contains:

- 📹 **25,335** video clips
- 🔊 **86** fine-grained event categories
- ⏱️ **Frame-level annotations** for precise localization

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your_username/AVE-PM.git
cd AVE-PM

conda env create -f environment.yaml
conda activate avepm
```



## 📁 Dataset

After downloading the PM400 dataset from [OneDrive Link](https://1drv.ms/f/c/8d9d5fbede2ace9d/Ep3OKt6-X50ggI2MAAAAAAABV0VlHe1CPMEbHIJ1ytZYZA?e=d1LJkF), you need to use our script to split the videos.

```python
python utils/cut_video_files.py
```

🔉 **Feature Extraction**

Audio feature and visual feature are also released. Please put videos of AVEPM dataset into /dataset/data/videos/ folder and features into /dataset/feature/ folder before running the code.

To extract features yourself:

```
python utils/encode.py
```

You may also use your own feature extraction method if desired.



Before training the LAVISH model, you should extract video frames and raw audios, and putting the output into /dataset/data/video_frames and /dataset/data/raw_audios folder.

```
python /LAVISH/scripts/extract_frames.py
python /LAVISH/scripts/extract_audio.py
```

Store the output into the following directories:

- Frames → `/dataset/data/video_frames/`
- Audios → `/dataset/data/raw_audios/`

### 📂 Directory Layout

```graphql
AVE-PM/
├── dataset/
|	├── csvfiles/			   # csv files for training,validating and testing
│   ├── data/
│   │   ├── videos/            # Raw portrait-mode videos
│   │   ├── video_frames/      # Extracted RGB frames (generated)
|	|	├── processed_audios/  # Preprocessed audio segments(generated)
│   │   └── raw_audios/        # Extracted audio segments (generated)
│   └── feature/			  
│       └── features/         # Precomputed audio and visual features
```



## 🚀 Training & Evaluation

We provide training and inference scripts for four baseline models evaluated on AVE-PM:

### 1. AVEL

```python
cd CPSP
# Training
python main.py --config ./CPSP/config/avel.yaml

# Evaluation (set evaluate=True and resume="your_checkpoint")
python main.py --config your_test_yaml_path

```



### 2. CPSP (Cross-modal Pseudo Supervision with Patch-level Confidence)

```python
cd CPSP
# Training
python main.py --config ./CPSP/config/cpsp.yaml

# Evaluation
python main.py --config your_test_yaml_path

```

### 3. CMBS (Cross-modal Background Suppression)

```python
cd CMBS
# Training
bash supv_train.sh

# Evaluation
bash supv_test.sh  # Edit 'resume' path inside script

```

### 4. LAVISH

```python
cd LAVISH
# Training
bash train.sh

# Evaluation
bash test.sh  # Set model_path to your checkpoint
```

### ✅ Pretrained Checkpoints

You can download pretrained model checkpoints here:
 *(Coming Soon — add download table with links if available)*



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



