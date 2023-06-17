import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb, cv2
import random

from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
import audiomentations as Au


def process_annotation(Annotation):
    with open(Annotation) as f:
        lines = f.readlines()
    print("totally {} lines".format(len(lines)))

    anno_dict = {}

    for i in lines[1:]:
        i = i.split('&')
        anno_dict[i[1]] = i[0]

    label_list = set(i for i in anno_dict.values())
    labels_double_ref = {}
    for i, label in enumerate(label_list):
        labels_double_ref[i] = label
        labels_double_ref[label] = i

    return anno_dict, labels_double_ref



class x360_Basic_Dataset(Dataset):
    
    def get_img_info(self, img_path, img, byshortmax=224, label=None):
        
        H, W, C = img.shape
        
        if W < H:
            ratio = byshortmax / W    # 
  
        elif H <= W:
            ratio = byshortmax / W
            
        Wcrop = int(W * ratio)  # size
        Hcrop = int(H * ratio)
        
        if label == 'crop':
            Hcrop = byshortmax
            Wcrop = int(2 * byshortmax)
            
                        
        rnd_h = random.randint(0, max(0, H - Hcrop))  # loc
        rnd_w = random.randint(0, max(0, W - Wcrop))
    
    
        info = {"H":H, "W":W, 
                "Hcrop":Hcrop, "Wcrop":Wcrop, 
                "rnd_h":rnd_h, "rnd_w":rnd_w}

        return info        
                
    def crop_by_ratio(self, img, center=False, info=None):
            
        if center:
            crop = A.Compose([
                A.CenterCrop(height=info['Hcrop'], width=info['Wcrop'], always_apply=True),
            ])
            cropped = crop(image=img)['image']
            
        else:
            cropped = img[info['rnd_h']:info['rnd_h'] + info['Hcrop'], 
                          info['rnd_w']:info['rnd_w'] + info['Wcrop']]

        return cropped

    
    def handle_audio_sample(self, sample, rate):
        
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)

        sample = self.au_transform(samples=sample, sample_rate=22050)

        new_sample = sample[start_point: start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        spectrogram = self.spec_transform(spectrogram)
        
        return spectrogram
    
    def read_audio(self, audio_path):

        sample, rate = librosa.load(audio_path, offset=0.0, mono=False,
                                    sr = 16000)  

        left_spectrogram = self.handle_audio_sample(sample[0], rate)
        right_spectrogram = self.handle_audio_sample(sample[1], rate)
     
        return [left_spectrogram, right_spectrogram]

    def read_audio_tracking(self, at):
        # read in directional audio time delay
        return np.load(at)   # Shape: [250]
    
    def read_frames(self, frame_folder, label=None):
        frame_list = glob(os.path.join(frame_folder, '*.jpg'))
        
        if len(frame_list)<self.use_video_frames:
            for i in range(self.use_video_frames - len(frame_list)):
                
                frame_list.append(frame_list[-1])
                
            select_index = frame_list
        else:
            select_index = np.random.choice(len(frame_list),
                                        size=self.use_video_frames, replace=False)
            
        select_index.sort()

        # every frame
        for i in range(self.use_video_frames):
            img = cv2.imread(frame_list[i])  
            img = self.resize(image=img)["image"] 
            
            if i == 0:
                info = self.get_img_info(frame_list[i], img, byshortmax=self.final_img_size, label=label)
                images = torch.zeros((self.use_video_frames, 3, info['Hcrop'], info['Wcrop']))
                
            img = self.crop_by_ratio(img, info=info, 
                                     center=(self.mode == 'test'))
   
            img = self.transform(image=img)["image"]

            try:
                images[i] = img
            except:
                print(frame_list[i], "failed")
                
        images = images.permute(1, 0, 2, 3)
        return images



    def idx2data(self, idx, frame_list, audio_list):

        data = self.csvfile.iloc[idx]

        path = data['path']
        classnum = data['classnum']

        # has n cut videos
        frame_list = glob(os.path.join(path, frame_list))
        frame_list.sort()

        audio_list = glob(os.path.join(path, audio_list))
        audio_list.sort()

        return classnum, frame_list, audio_list



    def get_at_data(self, idx):
        data = self.csvfile.iloc[idx]
        path = data['path']
        
        at_list = glob(os.path.join(path, self._front_view_audio_tracking_list))
        at_list.sort()
        
        return at_list
    
    def read_data(self, frame, audio, select_cut_id, label=None):

        return self.read_frames(frame[select_cut_id], label), \
               self.read_audio(audio[select_cut_id])
            
               

    def __len__(self):
        return len(self.csvfile)



class x360Dataset(x360_Basic_Dataset):
    def __init__(self, args, csvfile, mode='train', use_video_frames=3,
                 use_360=False, use_front=True, use_clip=False):

        self.args = args
        self.mode = mode
        self.csvfile = pd.read_csv(csvfile)
        self.data_list = []
        self.cut_number = 6

        self.final_img_size = 256
        self.resize_img_size = 512

        # $Root/ (Inside_Ouside)/ Location(Label)/ Video_ID
        root = '/bask/projects/j/jiaoj-3d-vision/360XProject/Data'

        self._360cut_frames_list = "360/360_panoramic_cut/frames/*"
        self._360cut_audios_list = "360/360_panoramic_cut/audios/*.wav"

        self._frontcut_frames_list = "360/front_view_cut/frames/*"
        self._frontcut_audios_list = "360/front_view_cut/audios/*.wav"

        self._Clip_frames_list = "Snapchat/clip1/binocular_cut/frames/*"
        self._Clip_audios_list = "Snapchat/clip1/binocular_cut/audios/*.wav"

        self._front_view_audio_tracking_list = "360/front_view_cut/at/*/front_audio_tracking.npy"
     
        self.use_video_frames = use_video_frames

        self.use_360 = use_360
        self.use_front = use_front
        self.use_clip = use_clip

        self.resize = A.Compose([
                A.SmallestMaxSize(self.resize_img_size, interpolation= 3)
                ])   
        
        
        if self.mode == 'train':
            
            self.transform = A.Compose([

                A.HorizontalFlip(p=0.5),    
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, p=0.5),

                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                        contrast_limit=(-0.1, 0.1), p=0.5),
                A.ImageCompression(p=0.3),  #
                
                A.Affine(p=0.3),  
                ToTensorV2(), 
            ])   
            
            self.au_transform = Au.Compose([
                    Au.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    Au.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                    Au.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                    Au.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                ])
            
            self.spec_transform = Au.SpecCompose([
                        # Au.SpecChannelShuffle(p=0.5),
                        Au.SpecFrequencyMask(p=0.5),
                ])

        else:
            self.transform = A.Compose([
                ToTensorV2(),
            ])

            self.au_transform = Au.Compose([])
            self.spec_transform = Au.SpecCompose([])
                

    def __getitem__(self, idx):

        length_of_list = 100

        # Load Data list
        if self.use_360:
            cls, _360frames_list, _360audio_list = \
                self.idx2data(idx, self._360cut_frames_list, self._360cut_audios_list)
            length_of_list = np.min([length_of_list, len(_360frames_list), len(_360audio_list)])

        if self.use_front:
            cls, _frontframes_list, _frontaudio_list = \
                self.idx2data(idx, self._frontcut_frames_list, self._frontcut_audios_list)
            length_of_list = np.min([length_of_list, len(_frontframes_list), len(_frontaudio_list)])

        if self.use_clip:
            cls, _Clipframes_list, _Clipaudio_list = \
                self.idx2data(idx, self._Clip_frames_list, self._Clip_audios_list)
            length_of_list = np.min([length_of_list, len(_Clipframes_list), len(_Clipaudio_list)])

        Output_dict = {"cls": cls}

        if self.args.use_directional_audio:
            at_list = self.get_at_data(idx)
            length_of_list = np.min([length_of_list, len(at_list)])
            
            
        try:
            select_cut_id = np.random.randint(0, length_of_list)
        except:
            print("failed select cut id:", length_of_list, _Clipframes_list)
            
        if self.args.use_directional_audio:
            at = self.read_audio_tracking(at_list[select_cut_id])[:250]  # clip at 250            
            if len(at) < 255:
                at = np.pad(at, (0,255-len(at)), 'constant', constant_values=(0,0))

            Output_dict['at'] = np.expand_dims(at, 0)
            

        # select Data
        if self.use_360:
            f, a = self.read_data(_360frames_list, _360audio_list, select_cut_id)
            Output_dict["360"] = {"f": f, "a": a}
            # print("Load 360 Data Success!")

        if self.use_front:
            f, a = self.read_data(_frontframes_list, _frontaudio_list, select_cut_id)
            Output_dict["front"] = {"f": f, "a": a}

        if self.use_clip:
            f, a = self.read_data(_Clipframes_list, _Clipaudio_list, select_cut_id, label='crop')
            Output_dict["clip"] = {"f": f, "a": a}
            # print("Load clip Data Success!")

        return Output_dict


class x360DatasetTest(x360_Basic_Dataset):
    def __init__(self, args, csvfile, mode='test', use_video_frames=3,
                 use_360=False, use_front=True, use_clip=False):


        self.args = args
        self.mode = mode
        self.csvfile = pd.read_csv(csvfile)
        self.data_list = []
        self.cut_number = 6

        self.final_img_size = 256  #256
        self.resize_img_size = 512  #512

        root = '/bask/projects/j/jiaoj-3d-vision/360XProject/Data'

        self._360cut_frames_list = "360/360_panoramic_cut/frames/*"
        self._360cut_audios_list = "360/360_panoramic_cut/audios/*.wav"

        self._frontcut_frames_list = "360/front_view_cut/frames/*"
        self._frontcut_audios_list = "360/front_view_cut/audios/*.wav"

        self._Clip_frames_list = "Snapchat/clip1/binocular_cut/frames/*"
        self._Clip_audios_list = "Snapchat/clip1/binocular_cut/audios/*.wav"

        self._front_view_audio_tracking_list = "360/front_view_cut/at/*/front_audio_tracking.npy"
             
        self.use_video_frames = use_video_frames

        self.use_360 = use_360
        self.use_front = use_front
        self.use_clip = use_clip

        self.resize = A.Compose([
            A.SmallestMaxSize(self.resize_img_size, interpolation=3)
        ])

        if self.mode == 'train':

            self.transform = A.Compose([

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, p=0.5),

                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                           contrast_limit=(-0.1, 0.1), p=0.5),
                A.ImageCompression(p=0.3),  #

                A.Affine(p=0.3),
                ToTensorV2(),
            ])

            self.au_transform = Au.Compose([
                Au.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                Au.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                Au.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Au.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            ])

            self.spec_transform = Au.SpecCompose([
                # Au.SpecChannelShuffle(p=0.5),
                Au.SpecFrequencyMask(p=0.5),
            ])

        else:
            self.transform = A.Compose([
                ToTensorV2(),
            ])

            self.au_transform = Au.Compose([])
            self.spec_transform = Au.SpecCompose([])

    def __getitem__(self, idx):

        length_of_list = 100

        
        # Load Data list
        if self.use_360:
            cls, _360frames_list, _360audio_list = \
                self.idx2data(idx, self._360cut_frames_list, self._360cut_audios_list)

            length_of_list = np.min([length_of_list, len(_360frames_list), len(_360audio_list)])

        if self.use_front:
            cls, _frontframes_list, _frontaudio_list = \
                self.idx2data(idx, self._frontcut_frames_list, self._frontcut_audios_list)
            length_of_list = np.min([length_of_list, len(_frontframes_list), len(_frontaudio_list)])

        if self.use_clip:
            cls, _Clipframes_list, _Clipaudio_list = \
                self.idx2data(idx, self._Clip_frames_list, self._Clip_audios_list)
            length_of_list = np.min([length_of_list, len(_Clipframes_list), len(_Clipaudio_list)])

        

        if self.args.use_directional_audio:
            at_list = self.get_at_data(idx)
            length_of_list = np.min([length_of_list, len(at_list)])
        
        Output_dict = {"cls": cls, "length": length_of_list}
            
        for j in range(length_of_list):
            Output_dict[j] = {}
            # select Data
            if self.args.use_directional_audio:
                at = self.read_audio_tracking(at_list[j])[:250]  # clip at 250
                if len(at) < 255:
                    at = np.pad(at, (0,255-len(at)), 'constant', constant_values=(0,0))
                    
                # Output_dict['at'] = np.expand_dims(at, 0)
                Output_dict['at'] = np.expand_dims(at, 0)
            
            
            if self.use_360:
                f, a = self.read_data(_360frames_list, _360audio_list, j)
                Output_dict[j]["360"] = {"f": f, "a": a}
                # print("Load 360 Data Success!")

            if self.use_front:
                f, a = self.read_data(_frontframes_list, _frontaudio_list, j)
                Output_dict[j]["front"] = {"f": f, "a": a}

            if self.use_clip:
                f, a = self.read_data(_Clipframes_list, _Clipaudio_list, j)
                Output_dict[j]["clip"] = {"f": f, "a": a}
                # print("Load clip Data Success!")

        return Output_dict


if __name__ == "__main__":
    Annotation = '../360data/AVE_Dataset/Annotations.txt'
    with open(Annotation) as f:
        lines = f.readlines()
    print("totally {} lines".format(len(lines)))

    label_dict = {}

    for i in lines[1:]:
        i = i.split('&')
        label_dict[i[1]] = i[0]

    label_list = set(i for i in label_dict.values())
    print(label_list)

