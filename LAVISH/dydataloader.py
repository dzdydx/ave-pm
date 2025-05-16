import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from ipdb import set_trace
import pickle as pkl
import h5py

import soundfile as sf
import torchaudio
import torchvision
import glob


### VGGSound
from scipy import signal
import soundfile as sf
###

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop
from PIL import Image

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import warnings
warnings.filterwarnings('ignore')
from scripts.visual_operate import *
import torch.nn as nn
class Visual_transform(nn.Module):
    def __init__(self, short_side_range=(150, 300), target_size=(224, 224)):
        super(Visual_transform, self).__init__()
        self.short_side_range = short_side_range
        self.target_size = target_size
    
    def forward(self, x):
        x = adjust_short(x, target_size=self.target_size, short_side_range=self.short_side_range)
        x = random_crop(x)
        return x


class DY_dataset(Dataset):
    def __init__(self, opt, ave=False, avepm=False, mode='train'):
        self.opt = opt
        assert not (ave and avepm), "only one of ave and avepm can be True"
        if ave:
            self.root = "/data1/yyp/AVEL/generalization_exp_data/ave_selected"
        elif avepm:
            self.root = "/data1/yyp/AVEL/generalization_exp_data/pm_selected"
        
        
        if mode == 'train':
            self.raw_gt = pd.read_csv(os.path.join(self.root, "train_meta_data.csv"))
        elif mode == 'val':
            self.raw_gt = pd.read_csv(os.path.join(self.root, "val_meta_data.csv"))
        elif mode == 'test':
            self.raw_gt = pd.read_csv(os.path.join(self.root, "test_meta_data.csv"))


		### ---> yb calculate: AVE dataset
        if self.opt.vis_encoder_type == 'vit':
            self.norm_mean = -4.1426
            self.norm_std = 3.2001
		### <----
		
        elif self.opt.vis_encoder_type == 'swin':
            ## ---> yb calculate: AVE dataset for 192
            self.norm_mean =  -4.984795570373535
            self.norm_std =  3.7079780101776123
            ## <----
			

        if self.opt.vis_encoder_type == 'vit':
            self.my_normalize = Compose([
                # Resize([384,384], interpolation=Image.BICUBIC),
                Resize([224,224], interpolation=Image.BICUBIC),
                # Resize([192,192], interpolation=Image.BICUBIC),
                # CenterCrop(224),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif self.opt.vis_encoder_type == 'swin':
            if self.opt.visual_operation:
                self.my_normalize = Compose([
                    Visual_transform(),
                    Resize([192,192], interpolation=Image.BICUBIC),
                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
            else:
                self.my_normalize = Compose([
                    Resize([192,192], interpolation=Image.BICUBIC),
                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ])
            # self.my_normalize = Compose([
            #     Resize([192,192], interpolation=Image.BICUBIC),
            #     Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            # ])

    def _wav2fbank(self, filename, filename2=None, idx=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        

        ## yb: align ##
        if waveform.shape[1] > 16000*(self.opt.audio_length+0.1):
            sample_indx = np.linspace(0, waveform.shape[1] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
            waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)]
        ## align end ##


        if self.opt.vis_encoder_type == 'vit':
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
            # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=512, dither=0.0, frame_shift=1)
        elif self.opt.vis_encoder_type == 'swin':
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

        ########### ------> very important: audio normalized
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ### <--------
        if self.opt.vis_encoder_type == 'vit':
            target_length = int(1024 * (1/10)) ## for audioset: 10s
        elif self.opt.vis_encoder_type == 'swin':
            target_length = 192 ## yb: overwrite for swin

        # target_length = 512 ## 5s
        # target_length = 256 ## 2.5s
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda
    def __len__(self):
        return len(self.raw_gt)

    def __getitem__(self, idx):
        file_name = str(self.raw_gt.iloc[idx]['video_id'])

        ### ---> loading all audio frames
        total_audio = []
        for audio_sec in range(10):
            fbank, mix_lambda = self._wav2fbank(self.opt.audio_folder+'/'+file_name+ '.wav', idx=audio_sec)
            total_audio.append(fbank)
        total_audio = torch.stack(total_audio)  # total_audio.shape = [10, 192, 192]
        ### <----

        ### ---> video frame process 
        total_num_frames = len(glob.glob(self.opt.video_folder+'/'+file_name+'/*.jpg'))
        sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
        total_img = []
        for vis_idx in range(10):
            tmp_idx = sample_indx[vis_idx]
            tmp_img = torchvision.io.read_image(self.opt.video_folder+'/'+file_name+'/'+ str("{:06d}".format(tmp_idx))+ '.jpg')/255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)	# [10, 3, 192, 192]
        ### <---

        ### one hot labels
        with open(os.path.join(self.root, "features", file_name+'.pkl'), 'rb') as f:
            feat = pkl.load(f)
        
        return {'audio_spec': total_audio, 
                'GT': feat['onehot_labels'].astype(np.float32), 
                # 'audio_vgg': self.audio_features[real_idx],
                'image':total_img
        }

class Audio_Preprocess_Dataset(Dataset):
    def __init__(self, opt, ave=False, avepm=False, mode='train', audio_process_mode="NMF1", is_select=True):
        self.opt = opt
        self.token = ""

        if is_select:
            self.token = "video_id"
            self.metaroot = "/data1/cy/MTN/dataset/avepm/select/"
            self.root = "/data1/yyp/AVEL/generalization_exp_data/pm_selected/features/"
        elif not is_select:
            self.token = "sample_id"
            self.metaroot = "/data1/cy/MTN/dataset/avepm/full/"
            self.root = "/data1/lwy/ave-pm-dataset/data/new_features/"


        self.raw_gt = pd.read_csv(os.path.join(self.metaroot, "{}.csv".format(mode)))

        audio_processmeta = pd.read_csv(os.path.join("/dataset/csvfiles/", "{}_meta.csv".format(audio_process_mode)))
        self.audio_process_ids = audio_processmeta["video_id"].tolist()
        self.audio_process_mode = audio_process_mode


		### ---> yb calculate: AVE dataset
        if self.opt.vis_encoder_type == 'vit':
            self.norm_mean = -4.1426
            self.norm_std = 3.2001
		### <----
		
        elif self.opt.vis_encoder_type == 'swin':
            ## ---> yb calculate: AVE dataset for 192
            self.norm_mean =  -4.984795570373535
            self.norm_std =  3.7079780101776123
            ## <----
			

        if self.opt.vis_encoder_type == 'vit':
            self.my_normalize = Compose([
                # Resize([384,384], interpolation=Image.BICUBIC),
                Resize([224,224], interpolation=Image.BICUBIC),
                # Resize([192,192], interpolation=Image.BICUBIC),
                # CenterCrop(224),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif self.opt.vis_encoder_type == 'swin':
            if self.opt.visual_operation:
                self.my_normalize = Compose([
                    Visual_transform(),
                    Resize([192,192], interpolation=Image.BICUBIC),
                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
            else:
                self.my_normalize = Compose([
                    Resize([192,192], interpolation=Image.BICUBIC),
                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ])


    def _wav2fbank(self, filename, filename2=None, idx=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        

        ## yb: align ##
        if waveform.shape[1] > 16000*(self.opt.audio_length+0.1):
            sample_indx = np.linspace(0, waveform.shape[1] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
            waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)]
        ## align end ##


        if self.opt.vis_encoder_type == 'vit':
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
            # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=512, dither=0.0, frame_shift=1)
        elif self.opt.vis_encoder_type == 'swin':
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

        ########### ------> very important: audio normalized
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ### <--------
        if self.opt.vis_encoder_type == 'vit':
            target_length = int(1024 * (1/10)) ## for audioset: 10s
        elif self.opt.vis_encoder_type == 'swin':
            target_length = 192 ## yb: overwrite for swin

        # target_length = 512 ## 5s
        # target_length = 256 ## 2.5s
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda
    def __len__(self):
        return len(self.raw_gt)

    def __getitem__(self, idx):
        file_name = str(self.raw_gt.iloc[idx][self.token])

        ### ---> loading all audio frames
        total_audio = []
        for audio_sec in range(10):
            if not file_name in self.audio_process_ids:
                fbank, mix_lambda = self._wav2fbank(self.opt.audio_folder+'/'+file_name+ '.wav', idx=audio_sec)
                total_audio.append(fbank)
            else:
                fbank, mix_lambda = self._wav2fbank(f"/dataset/data/processed_audios/{self.audio_process_mode}/{file_name}.wav", idx=audio_sec)
                total_audio.append(fbank)
        total_audio = torch.stack(total_audio)  # total_audio.shape = [10, 192, 192]
        ### <----

        ### ---> video frame process 
        total_num_frames = len(glob.glob(self.opt.video_folder+'/'+file_name+'/*.jpg'))
        sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
        total_img = []
        for vis_idx in range(10):
            tmp_idx = sample_indx[vis_idx]
            tmp_img = torchvision.io.read_image(self.opt.video_folder+'/'+file_name+'/'+ str("{:06d}".format(tmp_idx))+ '.jpg')/255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)	# [10, 3, 192, 192]
        ### <---

        ### one hot labels
        with open(os.path.join(self.root, file_name+'.pkl'), 'rb') as f:
            feat = pkl.load(f)
        
        return {'audio_spec': total_audio, 
                'GT': feat['onehot_labels'].astype(np.float32), 
                # 'audio_vgg': self.audio_features[real_idx],
                'image':total_img
        }

class Inception(nn.Module):
    def __init__(self, window_size):
        super(Inception, self).__init__()
        self.window_size = window_size
        
    def forward(self, img):
        height, width  = img.shape[1], img.shape[2]
        whole_size = height * width
        random_ratio = random.uniform(0.08, 1.0)
        target_pixel = int(whole_size * random_ratio)
        height_width_ratio = random.uniform(3/4, 4/3)
        target_height = int((target_pixel / height_width_ratio) ** 0.5)
        target_width = int(target_height * height_width_ratio)
        if target_height > height:
            target_height = height
        
        if target_width > width:
            target_width = width
        start_x = random.randint(0, width - target_width)
        start_y = random.randint(0, height - target_height)
        resize_func = Resize((self.window_size, self.window_size), interpolation=Image.BICUBIC)
        img = F.crop(img, top=start_y, left=start_x, height=target_height, width=target_width)
        img = resize_func(img)
        return img

################################ ADD Transform Func #######################################
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image

class Resize_Pad(nn.Module):
    def __init__(self, target_size=192):
        super(Resize_Pad, self).__init__()
        self.target_size = target_size

    def forward(self, image):
        # w, h = image.size
        h, w = image.shape[1], image.shape[2]
        scale = self.target_size / max(w, h)  # 将长边缩放为 target_size
        new_w, new_h = int(w * scale), int(h * scale)
        image = TF.resize(image, (new_w, new_h), interpolation=InterpolationMode.BILINEAR)

        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        # padding = (
        #     pad_w // 2, pad_h // 2,
        #     pad_w - pad_w // 2, pad_h - pad_h // 2
        # )
        padding = (
            pad_h // 2, pad_w // 2,
            pad_h - pad_h // 2, pad_w - pad_w // 2
        )
        image = TF.pad(image, padding, fill=0, padding_mode='constant')
        return image


class ScaleDataset(DY_dataset):
    def __init__(self, opt, ave=False, avepm=False, mode='train', scale_mode = 'center_crop'):
        super().__init__(opt, ave, avepm, mode)
        if scale_mode == 'center_crop':
            self.my_normalize = Compose([
                Resize(192, interpolation=Image.BICUBIC),
                CenterCrop(192),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif scale_mode == 'inception':
            self.my_normalize = Compose([
                Inception(192),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif scale_mode == 'random_crop':
            self.my_normalize = Compose([
                Resize(192, interpolation=Image.BICUBIC),
                RandomCrop(192),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif scale_mode == 'full_padding':
            self.my_normalize = Compose([
                Resize_Pad(192),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

class ExceptBGMDataset(DY_dataset):
    def __init__(self, opt, ave=False, avepm=False, mode='train'):
        super().__init__(opt, ave, avepm, mode)
        self.mode = mode
        bgm_info = pd.read_csv("/data1/yyp/AVEL/generalization_exp_data/pm_selected/meta_data_selected_with_BGM_info.csv")
        self.bgm_dict = {}
        for _, item in bgm_info.iterrows():
            self.bgm_dict[int(item['video_id'])] = item['haveBGM']
    
    def __getitem__(self, idx):
        file_name = str(self.raw_gt.iloc[idx]['video_id'])

        ### ---> loading all audio frames
        total_audio = []
        for audio_sec in range(10):
            fbank, mix_lambda = self._wav2fbank(self.opt.audio_folder+'/'+file_name+ '.wav', idx=audio_sec)
            total_audio.append(fbank)
        total_audio = torch.stack(total_audio)  # total_audio.shape = [10, 192, 192]
        ### <----

        ### ---> video frame process 
        total_num_frames = len(glob.glob(self.opt.video_folder+'/'+file_name+'/*.jpg'))
        sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
        total_img = []
        for vis_idx in range(10):
            tmp_idx = sample_indx[vis_idx]
            tmp_img = torchvision.io.read_image(self.opt.video_folder+'/'+file_name+'/'+ str("{:06d}".format(tmp_idx))+ '.jpg')/255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)	# [10, 3, 192, 192]
        ### <---

        ### one hot labels
        with open(os.path.join(self.root, "features", file_name+'.pkl'), 'rb') as f:
            feat = pkl.load(f)
        
        if self.mode == 'train' and self.bgm_dict[int(file_name)] == 1:
            total_audio = torch.zeros_like(total_audio)
        return {'audio_spec': total_audio, 
                'GT': feat['onehot_labels'].astype(np.float32), 
                # 'audio_vgg': self.audio_features[real_idx],
                'image':total_img
        }
        


class ToTensor(object):

    def __call__(self, sample):
        if len(sample) == 2:
            audio = sample['audio']
            label = sample['label']
            return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(label)}
        else:
            audio = sample['audio']
            video_s = sample['video_s']
            video_st = sample['video_st']
            label = sample['label']
            return {'audio': torch.from_numpy(audio), 'video_s': torch.from_numpy(video_s),
                    'video_st': torch.from_numpy(video_st),
                    'label': torch.from_numpy(label)}