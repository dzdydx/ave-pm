import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_pth', type=str, default='/data1/yyp/AVEL/VGG/AVE_dataset/AVE_Dataset/AVE', help='path to video')
parser.add_argument('--save_pth', type=str, default='/data1/yyp/AVEL/LAVISH/AVE_Dataset/raw_audio', help='path to save audio')
args = parser.parse_args()

video_pth =  args.video_pth
sound_list = os.listdir(video_pth)
save_pth =  args.save_pth

for audio_id in sound_list:
    name = os.path.join(video_pth, audio_id)
    audio_name = audio_id[:-4] + '.wav'
    exist_lis = os.listdir(save_pth)
    if audio_name in exist_lis:
        print("already exist!")
        continue
    try:
        video = VideoFileClip(name)
        audio = video.audio
        audio.write_audiofile(os.path.join(save_pth, audio_name), fps=16000)
        print("finish video id: " + audio_name)
    except:
        print("cannot load ", name)