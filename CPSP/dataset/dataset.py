import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pickle
import os

class DY_Dataset(Dataset):
    def __init__(self, data_root, meta_csv_path, split='train'):
        super(DY_Dataset, self).__init__()
        self.split = split
        csv_path = os.path.join(meta_csv_path, self.split+".csv")
        self.meta_csv = pd.read_csv(csv_path)
        sample_ids = self.meta_csv['sample_id'].tolist()
        self.filelist = []
        for item in sample_ids:
            self.filelist.append(os.path.join(data_root, "{}.pkl".format(item)))
    
    def __getitem__(self, index):
        file_path = self.filelist[index]
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        video_features = data["video_features"]
        audio_features = data["audio_features"]
        labels = data["onehot_labels"]
        return video_features, audio_features, labels

    def __len__(self):
        return len(self.filelist)
    
class ExceptBGMDataset(Dataset):
    def __init__(self, data_root, meta_csv_path, split='train'):
        super(ExceptBGMDataset, self).__init__()
        self.split = split
        csv_path = os.path.join(meta_csv_path, self.split+"_meta_data.csv")
        self.meta_csv = pd.read_csv(csv_path)
        sample_ids = self.meta_csv['video_id'].tolist()
        self.filelist = []
        for item in sample_ids:
            self.filelist.append(os.path.join(data_root, "{}.pkl".format(item)))

        bgm_info = pd.read_csv("/data1/yyp/AVEL/generalization_exp_data/pm_selected/meta_data_selected_with_BGM_info.csv")
        self.bgm_dict = {}
        for _, item in bgm_info.iterrows():
            self.bgm_dict[int(item['video_id'])] = item['haveBGM']
        
    def __getitem__(self, index):
        file_path = self.filelist[index]
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        video_features = data["video_features"]
        audio_features = data["audio_features"]
        labels = data["onehot_labels"]
        
        vid = os.path.basename(file_path).split(".")[0]
        if self.split=="train" and self.bgm_dict[int(vid)] == 1:
            audio_features = np.zeros_like(audio_features)
        return video_features, audio_features, labels
    
    def __len__(self):
        return len(self.filelist)

class Audio_Preprocess_Dataset(Dataset):
    def __init__(self, data_root, meta_csv_path, split='train', audio_preprocess_mode="NMF2", token="video_id"):
        super(Audio_Preprocess_Dataset, self).__init__()
        self.split = split
        csv_path = os.path.join(meta_csv_path, self.split+".csv")
        self.meta_csv = pd.read_csv(csv_path)
        self.token = token
        self.data_root = data_root
        NMFmeta = pd.read_csv(os.path.join("/dataset/", "{}_meta.csv".format(audio_preprocess_mode)))
        self.NMF_ids = NMFmeta['video_id'].tolist()
        self.audio_preprocess_mode = audio_preprocess_mode

    def __getitem__(self, index):
        video_id = self.meta_csv.iloc[index][self.token]
        with open(os.path.join(self.data_root, "{}.pkl".format(video_id)), "rb") as f:
            data = pickle.load(f)
        video_features = data["video_features"]
        audio_features = data["audio_features"]
        labels = data["onehot_labels"]

        # if video_id in self.NMF_ids and self.split == "train":
        if video_id in self.NMF_ids:
            with open(os.path.join("/dataset/feature/preprocess_audio_feature/{}_features/".format(self.audio_preprocess_mode),  "{}.pkl".format(video_id)), "rb") as f:
                NMF_data = pickle.load(f)
            audio_features = NMF_data["audio_features"]
        
        return video_features, audio_features, labels
    
    def __len__(self):
        return len(self.meta_csv)

if __name__ == "__main__":
    # split the data
    meta_csv_path = ""
    meta_csv = pd.read_csv(meta_csv_path)
    # 打乱
    meta_csv = meta_csv.sample(frac=1)
    total_len = len(meta_csv)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len
    train_csv = meta_csv[:train_len]
    val_csv = meta_csv[train_len:train_len+val_len]
    test_csv = meta_csv[train_len+val_len:]
    train_csv.to_csv("train.csv", index=False)
    val_csv.to_csv("val.csv", index=False)
    test_csv.to_csv("test.csv", index=False)