import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np

class AVEDataset(Dataset):
    def __init__(self, data_root, meta_root,split='train', ave=False, avepm=False, preprocess_mode='None', audio_process_mode="None", is_select=False, 
                 v_feature_root="dataset/feature/preprocess_visual_feature",
                 a_feature_root="dataset/feature/preprocess_audio_feature"):
        super(AVEDataset, self).__init__()
        self.split = split
        assert not (ave and avepm), "enable two datsets at the same time"
        self.ave = ave
        self.avepm = avepm

        self.preprocess_mode = preprocess_mode
        self.audio_process_mode = audio_process_mode
        self.is_select = is_select
        self.a_feature_root = a_feature_root
        self.v_feature_root = v_feature_root
        self.data_root = data_root
        self.meta_root = meta_root

        if self.audio_process_mode != 'None':   
            audio_processmeta = pd.read_csv(os.path.join(self.meta_root, "{}_meta.csv".format(audio_process_mode)))
            self.audio_process_ids = audio_processmeta["sample_id"].tolist()

        if self.is_select:
            if ave:
                self.meta_root = os.path.join(meta_root, 'select', 'ave')
            elif avepm:
                self.meta_root = os.path.join(meta_root, 'select', 'avepm')


        self.raw_gt = pd.read_csv(os.path.join(self.meta_root, "{}.csv".format(split)))


    def __getitem__(self, index):
        sample_id = self.raw_gt.iloc[index]['sample_id']
        with open(os.path.join(self.data_root, f"{sample_id}.pkl"), 'rb') as f:
            data = pickle.load(f)
        video_features = data["video_features"]
        audio_features = data["audio_features"]
        labels = data["onehot_labels"]

        if self.preprocess_mode != 'None':
            with open(os.path.join(self.v_feature_root, f"{sample_id}.pkl"), 'rb') as f:
                v_data= pickle.load(f)
            video_features = v_data[self.preprocess_mode]
        
        if self.audio_process_mode != 'None':
            if sample_id in self.audio_process_ids and self.split == 'train':
                with open(os.path.join(self.a_feature_root, f"{self.audio_process_mode}", f"{sample_id}.pkl"), 'rb') as f:
                    audio_process_data = pickle.load(f)
                audio_features = audio_process_data["audio_features"]
        
        return video_features, audio_features, labels

    def __len__(self):
        return len(self.raw_gt)