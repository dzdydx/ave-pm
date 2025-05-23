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
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
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
        padding = (
            pad_h // 2, pad_w // 2,
            pad_h - pad_h // 2, pad_w - pad_w // 2
        )
        image = TF.pad(image, padding, fill=0, padding_mode='constant')
        return image

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
        
class AVELDataset(Dataset):
    def __init__(self, opt, data_root, meta_root,split='train', 
                 ave=False, avepm=False, 
                 preprocess_mode='None', audio_process_mode="None", 
                 processed_audio_root="dataset/data/processed_audios/",
                 is_select=False):
        super(AVELDataset, self).__init__()
        self.split = split
        self.opt = opt
        assert not (ave and avepm), "enable two datsets at the same time"
        self.ave = ave
        self.avepm = avepm

        self.preprocess_mode = preprocess_mode
        self.audio_process_mode = audio_process_mode
        self.is_select = is_select
        self.data_root = data_root
        self.meta_root = meta_root
        self.processed_audio_root = processed_audio_root

        if self.audio_process_mode != 'None':   
            audio_processmeta = pd.read_csv(os.path.join(self.meta_root, "{}_meta.csv".format(audio_process_mode)))
            self.audio_process_ids = audio_processmeta["sample_id"].tolist()


        if self.is_select:
            if ave:
                self.meta_root = os.path.join(meta_root, 'select', 'ave')
            elif avepm:
                self.meta_root = os.path.join(meta_root, 'select', 'avepm')

        self.raw_gt = pd.read_csv(os.path.join(self.meta_root, "{}.csv".format(split)))
        ## ---> yb calculate: AVE dataset for 192
        self.norm_mean =  -4.984795570373535
        self.norm_std =  3.7079780101776123
            ## <----
		
        if preprocess_mode == 'None':
            self.my_normalize = Compose([
                    Resize([192,192], interpolation=Image.BICUBIC),
                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ])
        elif preprocess_mode == 'center_crop':
            self.my_normalize = Compose([
                Resize(192, interpolation=Image.BICUBIC),
                CenterCrop(192),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif preprocess_mode == 'inception':
            self.my_normalize = Compose([
                Inception(192),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif preprocess_mode == 'random_crop':
            self.my_normalize = Compose([
                Resize(192, interpolation=Image.BICUBIC),
                RandomCrop(192),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif preprocess_mode == 'longer_side_resize':
            self.my_normalize = Compose([
                Resize_Pad(192),
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
    
    def __getitem__(self, index):
        sample_id = str(self.raw_gt.iloc[index]['sample_id'])
        ### ---> loading all audio frames
        total_audio = []
        for audio_sec in range(10):
            if self.audio_process_mode == 'None' or self.split != 'train':
                fbank, mix_lambda = self._wav2fbank(self.opt.audio_folder+'/'+sample_id+ '.wav', idx=audio_sec)
                total_audio.append(fbank)
            else:
                fbank, mix_lambda = self._wav2fbank(f"{self.processed_audio_root}/{self.audio_process_mode}/{sample_id}.wav", idx=audio_sec)
                total_audio.append(fbank)
        total_audio = torch.stack(total_audio)  # total_audio.shape = [10, 192, 192]
        ### <----

        ### ---> video frame process 
        total_num_frames = len(glob.glob(self.opt.video_folder+'/'+sample_id+'/*.jpg'))
        sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
        total_img = []
        for vis_idx in range(10):
            tmp_idx = sample_indx[vis_idx]
            tmp_img = torchvision.io.read_image(self.opt.video_folder+'/'+sample_id+'/'+ str("{:06d}".format(tmp_idx))+ '.jpg')/255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)	# [10, 3, 192, 192]
        ### <---
        
        ### one hot labels
        with open(os.path.join(self.data_root, sample_id+'.pkl'), 'rb') as f:
            feat = pkl.load(f)
        
        return {'audio_spec': total_audio, 
                'GT': feat['onehot_labels'].astype(np.float32), 
                # 'audio_vgg': self.audio_features[real_idx],
                'image':total_img
        }


        