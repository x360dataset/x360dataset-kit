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

import models.vggish_src.mel_features as mel_features
import models.vggish_src.vggish_params as vggish_params
import numpy as np
import resampy
import soundfile as sf


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

    def waveform_to_examples(self, data, sample_rate, return_tensor=True):
        """Converts audio waveform into an array of examples for VGGish.

    Args:
        data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
        sample_rate: Sample rate of data.
        return_tensor: Return data as a Pytorch tensor ready for VGGish

    Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which represents
        a sequence of examples, each of which contains a patch of log mel
        spectrogram, covering num_frames frames of audio and num_bands mel frequency
        bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.

    """
        # Convert to mono.
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        # Resample to the rate assumed by VGGish.
        if sample_rate != vggish_params.SAMPLE_RATE:
            data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

        # Compute log mel spectrogram features.
        log_mel = mel_features.log_mel_spectrogram(
            data,
            audio_sample_rate=vggish_params.SAMPLE_RATE,
            log_offset=vggish_params.LOG_OFFSET,
            window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
            hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
            num_mel_bins=vggish_params.NUM_MEL_BINS,
            lower_edge_hertz=vggish_params.MEL_MIN_HZ,
            upper_edge_hertz=vggish_params.MEL_MAX_HZ)

        # Frame features into examples.
        features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
        example_window_length = int(round(
            vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        example_hop_length = int(round(
            vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
        log_mel_examples = mel_features.frame(
            log_mel,
            window_length=example_window_length,
            hop_length=example_hop_length)
        
        return log_mel_examples


    def handle_audio_sample(self, sample, rate):

        start_point = random.randint(a=0, b=np.max([0, len(sample) - rate]))

        sample = self.au_transform(samples=sample, sample_rate=rate)
        sample[sample > 1.] = 1.
        sample[sample < -1.] = -1.

        tuple_spec = []

        l = sample.shape[0] // self.tuple_len   

        sample = np.concatenate([sample, sample])

        for i in range(self.tuple_len):
            new_sample = sample[start_point: start_point + l]  

            start_point = start_point + l

            spectrogram = self.waveform_to_examples(new_sample, sample_rate=rate)[:5]

            if spectrogram.shape[0] < 5:
                s = spectrogram.shape[0]
                try:
                    ratio = (5 // s + 1) // 2 + 1
                except:
                    print("failed to get ratio:",
                          spectrogram.shape, new_sample.shape, sample.shape)

                for i in range(ratio):
                    spectrogram = np.concatenate([spectrogram, spectrogram], axis=0)

            spectrogram = self.spec_transform(spectrogram)

            tuple_spec.append(torch.from_numpy(spectrogram[:5]))


        return tuple_spec


    def read_audio(self, audio_path, sr_in):

        base_sr = 16000 // sr_in
        samples, sr = librosa.load(audio_path, offset=0.0, mono=False, sr=base_sr)  # sr=16000,   - > 220618,)


        left_spectrogram = self.handle_audio_sample(samples[0], sr)
        right_spectrogram = self.handle_audio_sample(samples[:, 1], sr)


        return [left_spectrogram, right_spectrogram, audio_path]



    def read_audio_tracking(self, at):
        # read in directional audio time delay
        return np.load(at) # [250]
    
    def read_frames(self, frame_folder, clip_sr=1, label=None):

        tuple_clip = []

        tuple_start = random.randint(0, 36)  # 64
        clip_start = tuple_start

        frame_list = glob(os.path.join(frame_folder, '*.jpg'))
        frame_list.sort()
        
        num_frames = len(frame_list)
            
        try:
            ratio = int(np.ceil(self.clip_len / len(frame_list))) + 1
        except:
            print("frame_folder:", frame_folder)

        cache_frame_list = frame_list.copy()
        for i in range(ratio):

            frame_list.extend(cache_frame_list)

        self.transforms = torch.nn.Sequential(
                transforms.RandomVerticalFlip(p= (1.0 if np.random.random() > 0.5 else 0) ),
                transforms.RandomHorizontalFlip(p=(1.0 if np.random.random() > 0.5 else 0))
            )

        idx = 0
        tuble_id_list = []
        
        for _ in range(self.tuple_len):
            clip_id_list = []
            
            for i in range(self.clip_len):
                # clip_sr is the pace of video pace, usually randomly sampled from 1~4 
                     
                if (clip_start + (idx + 1) * clip_sr) >= num_frames:
                    clip_start = (clip_start + (idx + 1)* clip_sr) - num_frames
                    idx = 0
                    
                try:
                    img = cv2.imread(frame_list[clip_start + idx * clip_sr])
                    clip_id_list.append(clip_start + idx * clip_sr)
                    
                except:
                    print(f"frame {clip_start + idx * clip_sr} of {num_frames} and len {len(frame_list)}")
                    
                img = self.resize(image=img)["image"]

                if i == 0:
                    info = self.get_img_info(frame_list[i], img, byshortmax=self.final_img_size, label=label)
                    images = torch.zeros((self.clip_len, 3, info['Hcrop'], info['Wcrop']))

                img = self.crop_by_ratio(img, info=info,
                                         center=(self.mode == 'test'))

                img = self.transform(image=img)["image"]

                images[i] = img

                idx += 1

            images = images.permute(1, 0, 2, 3) 

           
            images = self.transforms(images)

            tuple_clip.append(images)
            tuble_id_list.append(clip_id_list)
            
            clip_start = clip_start + self.clip_len 

        return tuple_clip

    
    # Frame Order Shuffle
    def order_shuffle(self, tuple_clip, tuple_audio):

        tuple_order = list(range(0, self.tuple_len))

        clip_and_order = list(zip(tuple_clip, tuple_audio, tuple_order))
        random.shuffle(clip_and_order)
        tuple_clip, tuple_audio, tuple_order = zip(*clip_and_order)

        return torch.stack(tuple_clip), torch.cat(tuple_audio), \
               torch.tensor(tuple_order)              # np.concatenate(tuple_audio), \


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
    
    def read_data(self, frame, audio, select_cut_id, clip_sr=1, label=None):
        
        # clip_sr is the pace of video pace, usually randomly sampled from 1~4
        
        return self.read_frames(frame[select_cut_id],  label=label, clip_sr=clip_sr), \
               self.read_audio(audio[select_cut_id], clip_sr)

    def __len__(self):
        return len(self.csvfile)


class x360Dataset(x360_Basic_Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
        Need the corresponding configuration file exists.

        Args:
            root_dir (string): Directory with videos and splits.
            train (bool): train split or test split.
            clip_len (int): number of frames in clip, 16/32/64.
            interval (int): number of frames between clips, 16/32.
            tuple_len (int): number of clips in each tuple, 3/4/5.
            transforms_ (object): composed transforms which takes in PIL image and output tensors.
        """

    def __init__(self, args, csvfile, mode='train',
                 use_360=False, use_front=True, use_clip=False):

        self.args = args
        self.clip_len = args.cl   # length of video clip  
        self.max_sr = args.max_sr          # max sampling rate

        self.interval = args.it   # args.cl, args.it, args.tl
        self.tuple_len = args.tl  # number of clips in each tuple, 3/4/5.

        self.mode = mode
        self.csvfile = pd.read_csv(csvfile)
        self.data_list = []
        self.cut_number = 6

        self.final_img_size = 128  #args.crop_sz
        self.resize_img_size = 512

        # $Root/ (Inside_Ouside)/ Location(Label)/ Video_ID
        root = '/bask/projects/j/jiaoj-3d-vision/360XProject/Data'

        self._360cut_frames_list = "360/360_panoramic_cut/all_frames/*"
        self._360cut_audios_list = "360/360_panoramic_cut/audios/*.wav"

        self._frontcut_frames_list = "360/front_view_cut/all_frames/*"
        self._frontcut_audios_list = "360/front_view_cut/audios/*.wav"

        self._Clip_frames_list = "Snapchat/clip1/binocular_cut/all_frames/*"
        self._Clip_audios_list = "Snapchat/clip1/binocular_cut/audios/*.wav"

        self._front_view_audio_tracking_list = "360/front_view_cut/at/*/front_audio_tracking.npy"

        self.use_360 = use_360
        self.use_front = use_front
        self.use_clip = use_clip

        self.resize = A.Compose([
                A.SmallestMaxSize(self.resize_img_size, interpolation=3)
                ])   

        if self.mode == 'train':
            
            self.transform = A.Compose([
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, p=0.5),
                
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                           contrast_limit=(-0.1, 0.1), p=0.5),
                A.ImageCompression(p=0.3),  #
                
                # A.Affine(p=0.3),
                ToTensorV2(), 
            ])   
            
            # , keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)

            self.au_transform = Au.Compose([
                    Au.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    Au.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                    Au.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                    Au.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                ])
            
            self.spec_transform = Au.SpecCompose([
                        # Au.SpecChannelShuffle(p=0.5),
                        Au.SpecFrequencyMask(p=0.5),
                        # ToTensorV2()
                ])

        else:
            self.transform = A.Compose([
                ToTensorV2(),
            ])

            self.au_transform = Au.Compose([])
            self.spec_transform = Au.SpecCompose([])
                

    def __getitem__(self, idx):

        length_of_list = 100
        clip_sr = random.randint(1, self.max_sr)
        

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
   
        Output_dict = {"cls": cls, "sr": clip_sr}

        if self.args.use_directional_audio:
            at_list = self.get_at_data(idx)
            length_of_list = np.min([length_of_list, len(at_list)])


        select_cut_id = np.random.randint(0, length_of_list)   # number of clips

        if self.args.use_directional_audio:
            at = self.read_audio_tracking(at_list[select_cut_id])[:250]  # clip at 250            
            if len(at) < 255:
                at = np.pad(at, (0,255-len(at)), 'constant', constant_values=(0, 0))

            Output_dict['at'] = np.expand_dims(at, 0)
            

        # select Data
        if self.use_360:
            # tuple
            f, a = self.read_data(_360frames_list, _360audio_list, select_cut_id,
                                  clip_sr=clip_sr)
            f, a, o = self.order_shuffle(f, a[0])

            Output_dict["360"] = {"f": f, "order": o, "a": a}

        if self.use_front:
            f, a = self.read_data(_frontframes_list, _frontaudio_list, select_cut_id,
                                   clip_sr=clip_sr)

            f, a, o = self.order_shuffle(f, a[0])

            Output_dict["front"] = {"f": f, "order": o, "a": a}

        if self.use_clip:
            f, a = self.read_data(_Clipframes_list, _Clipaudio_list, select_cut_id,
                                  label='crop', clip_sr=clip_sr)
            f, a, o = self.order_shuffle(f, a[0])

            Output_dict["clip"] = {"f": f, "order": o, "a": a}

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

