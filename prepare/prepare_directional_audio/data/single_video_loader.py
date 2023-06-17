import csv
import glob
import h5py
import io
import json
import librosa
import numpy as np
import os
import pickle
from PIL import Image
from PIL import ImageFilter
import random
import scipy
import soundfile as sf
import time
from tqdm import tqdm
import glob
import cv2

import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms

import sys
sys.path.append('..')
from utils import sound, sourcesep
from data import * 

import pdb


class SingleVideoDataset(StereoAudioDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        self.pr = pr
        self.split = split
        self.seed = pr.seed
        self.aug_wave = args.aug_wave
        self.no_resample = args.no_resample
        self.wav2spec = args.wav2spec
        # self.pr.clip_length = 0.96 if self.wav2spec else self.pr.clip_length
        self.fake_right = args.fake_right
        self.synthetic_rate = args.synthetic_rate if split == 'train' else 0.0
        self.normalized_rms = args.normalized_rms
        self.add_noise_with_snr = args.noiseSNR

        # save args parameter
        self.repeat = args.repeat if split == 'train' else 1
        # self.class_info = pr.class_info
        self.image_transform = transforms.Compose(self.generate_image_transform(args, pr))
        self.video_transform = transforms.Compose(self.generate_video_transform(args, pr))
        
        # self.list_sample = self.get_list_sample(list_sample)
        self.data_weight = self.get_data_weight()
        if split in ['train', 'test']:
            pr.data_weight = self.data_weight

        video_path = list_sample
        video_name = video_path.split("/")[-1]
        video_path_noname = video_path.rstrip(video_name).rstrip("/")

        audio_path = os.path.join(video_path_noname, 'at', video_name.rstrip(".mp4"), 'audio', 'audio.wav')
        frame_path = os.path.join(video_path_noname, 'at', video_name.rstrip(".mp4"), 'frames')

        meta_path = os.path.join(video_path_noname, 'at', video_name.rstrip(".mp4"), 'meta', video_name.rstrip('.mp4')+'_meta.json')  # meta...
        with open(meta_path, "r") as f:
            self.meta_dict = json.load(f)
        
        # audio_sample_rate = meta_dict['audio_sample_rate']
        # frame_rate = meta_dict['frame_rate']
        
        frame_list = glob.glob(f'{frame_path}/*.jpg')
        frame_list.sort()
        
        self.frame_list = frame_list
        
        audio, self.audio_rate = self.read_audio(audio_path)
        self.audio = torch.from_numpy(audio.copy()).float()
        num_sample = len(self.frame_list)
        
        # self.class_dist = self.unbalanced_dist()
        print('Video Dataloader: # sample of {}: {}'.format(self.split, num_sample))


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        audio_length = self.audio.shape[1]
        frame_path = self.frame_list[index]       # could be vacant

        start_time = index / self.meta_dict['frame_rate'] - self.pr.clip_length / 2
        audio_rate = self.audio_rate
        offset = int(self.pr.patch_num * self.pr.patch_stride)
        clip_length = int(self.pr.clip_length * self.audio_rate + offset)
        audio_start_time = int(start_time * self.audio_rate)
        audio_end_time = audio_start_time + clip_length

        if audio_start_time < 0:
            audio_start_time = 0
            audio_end_time = audio_start_time + clip_length

        if audio_end_time > audio_length:
            audio_end_time = audio_length
            audio_start_time = audio_end_time - clip_length
        
        imgs = self.read_image([frame_path])
        audio = self.audio[:, audio_start_time: audio_end_time]
        
        if_fakeright = np.random.rand() < 0

        patch_size = int(self.pr.clip_length * audio_rate)
        lefts, rights, audio = self.generate_audio(audio, audio_rate, if_fakeright, index, patch_size)

        delay_time = self.create_delay_matrix(self.pr.patch_num + 1, audio_rate)

        batch = {
            'img': imgs,
            'img_path': frame_path,
            'audio': audio,
            'left_audios': lefts,
            'right_audios': rights,
            'audio_rate': torch.tensor(audio_rate),
            'delay_time': delay_time
        }
        return batch

    def getitem_test(self, index):
        self.__getitem__(index)

    def __len__(self): 
        return len(self.frame_list)

    def generate_audio(self, audio, audio_rate, if_fakeright, index, patch_size=None):
        # import pdb; pdb.set_trace()
        patch_size = self.pr.patch_size if patch_size is None else patch_size
        audio_length = audio.shape[1]
        lefts = audio[0:1, :]
        rights = audio[1:2, :]
        return lefts, rights, audio