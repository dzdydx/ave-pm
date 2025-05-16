import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'labels.h5')
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True
        sample_index = self.sample_order[index]
        visual_feat = self.visual_feature[sample_index]
        audio_feat = self.audio_feature[sample_index]
        label = self.labels[sample_index]

        return visual_feat, audio_feat, label


    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num

import pandas as pd
import pickle as pkl
import numpy as np

class DouyinDataset(Dataset):
    def __init__(self, split='train', ave=False, avepm=False):
        super(DouyinDataset, self).__init__()
        
        assert not (ave and avepm), "enable two datsets at the same time"
        if ave:
            root = "/data1/yyp/AVEL/generalization_exp_data/ave_selected"
        elif avepm:
            root = "/data1/yyp/AVEL/generalization_exp_data/pm_selected"
        
        self.feature_path = os.path.join(root, "features")

        if split == 'train':
            self.raw_gt = pd.read_csv(os.path.join(root, "train_meta_data.csv"))
        elif split == 'val':
            self.raw_gt = pd.read_csv(os.path.join(root, "val_meta_data.csv"))
        elif split == 'test':
            self.raw_gt = pd.read_csv(os.path.join(root, "test_meta_data.csv"))
        
        # with open("/data1/yyp/AVEL/LAVISH/Douyin_Dataset/final_labels.pkl", "rb") as f:
        #     self.labels = pkl.load(f)
    
    def __len__(self):
        return len(self.raw_gt)

    def __getitem__(self, index):
        sample_id = self.raw_gt.iloc[index]['video_id']
        # real_index = self.raw_gt.iloc[index]['index']
        with open(os.path.join(self.feature_path, f"{sample_id}.pkl"), "rb") as f:
            features = pkl.load(f)
        return features['video_features'], features['audio_features'], features['onehot_labels']
    
from tqdm import tqdm
class ScaleDataset(Dataset):
    def __init__(self, split='train', ave=False, avepm=False, preprocess_mode='center_crop'):
        super(ScaleDataset, self).__init__()
        self.split = split

        assert not (ave and avepm), "enable two datsets at the same time"
        if ave:
            root = "/data1/yyp/AVEL/generalization_exp_data/ave_selected"
        elif avepm:
            root = "/data1/yyp/AVEL/generalization_exp_data/pm_selected"
        self.feature_path = os.path.join(root, "features")

        if split == 'train':
            self.raw_gt = pd.read_csv(os.path.join(root, "train_meta_data.csv"))
        elif split == 'val':
            self.raw_gt = pd.read_csv(os.path.join(root, "val_meta_data.csv"))
        elif split == 'test':
            self.raw_gt = pd.read_csv(os.path.join(root, "test_meta_data.csv"))
        sample_ids = self.raw_gt['video_id'].tolist()
        # self.filelist = []
        # self.vision_filelist = []
        # for item in sample_ids:
        #     self.filelist.append(os.path.join(data_root, "{}.pkl".format(item)))
        #     self.vision_filelist.append(os.path.join("/data1/yyp/AVEL/VGG/new_features/feature1", "{}.pkl".format(item)))

        self.features = []
        for item in tqdm(sample_ids, desc="Loading data"):
            tmp = {}
            with open(os.path.join(self.feature_path, "{}.pkl".format(item)), "rb") as f:
                data = pkl.load(f)
                tmp["audio_features"] = data["audio_features"]
                tmp["onehot_labels"] = data["onehot_labels"]
            # with open(os.path.join("/data1/yyp/AVEL/VGG/new_features/feature2", "{}.pkl".format(item)), "rb") as f:
            with open(os.path.join("/data1/cy/MTN/feature/full_img/", "{}.pkl".format(item)), "rb") as f:
                vision_data = pkl.load(f)
                # tmp["video_features"] = vision_data[preprocess_mode].cpu().numpy()
                tmp["video_features"] = vision_data["video_features"]
            self.features.append(tmp)
    
    def __getitem__(self, index):
        # file_path = self.filelist[index]
        # with open(file_path, "rb") as f:
        #     data = pickle.load(f)
        # # video_features = data["video_features"]
        # audio_features = data["audio_features"]
        # labels = data["onehot_labels"]

        # vision_file_path = self.vision_filelist[index]
        # with open(vision_file_path, "rb") as f:
        #     vision_data = pickle.load(f)
        # video_features = vision_data["center_crop"]
        features = self.features[index]

        return features['video_features'], features['audio_features'], features['onehot_labels']
    
    def __len__(self):
        return len(self.features)
    

import pickle

class Audio_Preprocess_Dataset(Dataset):
    def __init__(self, is_select = False, split='train', audio_process_mode="NMF2"):
        super(Audio_Preprocess_Dataset, self).__init__()
        self.split = split
        self.token = ""

        if is_select:
            self.token = "video_id"
            self.root = "/data1/cy/MTN/dataset/avepm/select/"
            self.feature_path = "/data1/yyp/AVEL/generalization_exp_data/pm_selected/features/"
        elif not is_select:
            self.token = "sample_id"
            self.root = "/dataset/csvfiles/"
            self.feature_path = "/dataset/feature/new_features/"
        
        self.raw_gt = pd.read_csv(os.path.join(self.root, "{}.csv".format(split)))

        audio_processmeta = pd.read_csv(os.path.join("/dataset/csvfiles/", "{}_meta.csv".format(audio_process_mode)))
        self.audio_process_ids = audio_processmeta["video_id"].tolist()
        self.audio_process_mode = audio_process_mode

    def __getitem__(self, index):
        video_id = self.raw_gt.iloc[index][self.token]
        with open(os.path.join(self.feature_path, f"{video_id}.pkl"), "rb") as f:
            data = pickle.load(f)
        video_features = data["video_features"]
        audio_features = data["audio_features"]
        labels = data["onehot_labels"]

        if video_id in self.audio_process_ids and self.split=="train":
            with open(os.path.join("/dataset/feature/preprocess_audio_feature/{}_features/".format(self.audio_process_mode),  "{}.pkl".format(video_id)), "rb") as f:
                audio_process_data = pickle.load(f)
            audio_features = audio_process_data["audio_features"]
        
        return video_features, audio_features, labels
    
    def __len__(self):
        return len(self.raw_gt)
    
class ExceptBGMDataset(Dataset):
    def __init__(self, split='train', ave=False, avepm=False):
        super(ExceptBGMDataset, self).__init__()
        self.split = split
        assert not (ave and avepm), "enable two datsets at the same time"
        if ave:
            root = "/data1/yyp/AVEL/generalization_exp_data/ave_selected"
        elif avepm:
            root = "/data1/yyp/AVEL/generalization_exp_data/pm_selected"
        
        self.feature_path = os.path.join(root, "features")

        if split == 'train':
            self.raw_gt = pd.read_csv(os.path.join(root, "train_meta_data.csv"))
        elif split == 'val':
            self.raw_gt = pd.read_csv(os.path.join(root, "val_meta_data.csv"))
        elif split == 'test':
            self.raw_gt = pd.read_csv(os.path.join(root, "test_meta_data.csv"))
        
        bgm_info = pd.read_csv("/data1/yyp/AVEL/generalization_exp_data/pm_selected/meta_data_selected_with_BGM_info.csv")
        self.bgm_dict = {}
        for _, item in bgm_info.iterrows():
            self.bgm_dict[int(item['video_id'])] = item['haveBGM']
    
    def __len__(self):
        return len(self.raw_gt)

    def __getitem__(self, index):
        sample_id = self.raw_gt.iloc[index]['video_id']
        # real_index = self.raw_gt.iloc[index]['index']
        with open(os.path.join(self.feature_path, f"{sample_id}.pkl"), "rb") as f:
            features = pkl.load(f)
        if self.split=="train" and self.bgm_dict[int(sample_id)] == 1:
            features['audio_features'] = np.zeros_like(features['audio_features'])
        return features['video_features'], features['audio_features'], features['onehot_labels']