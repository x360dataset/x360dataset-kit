from .models.extract_i3d import ExtractI3D
from omegaconf import OmegaConf
import torch
import os
from pathlib import Path
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from transformers import VideoMAEConfig, VideoMAEModel, VideoMAEFeatureExtractor
from transformers import AutoImageProcessor
import av

configuration = VideoMAEConfig()

# model = VideoMAEFeatureExtractor(configuration)
# configuration = model.config

def read_video_pyav(container, indices):
    
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).

    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, current_start, seg_len, frame_sample_rate=1):

    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices

    '''

    converted_len = int(clip_len * frame_sample_rate)

    end_idx = current_start + converted_len #seg_len  #np.random.randint(converted_len, seg_len)

    start_idx = current_start   #end_idx - converted_len

    indices = np.linspace(start_idx, end_idx, num=clip_len)

    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

    return indices

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model.cuda()

step_size = 1
clip_len = 16

def get_feature(file_path):
    # video clip consists of 300 frames (10 seconds at 30 FPS)
    container = av.open(file_path)

    current_start = 0
    total_frames = container.streams.video[0].frames
    
    while (current_start < total_frames):
        # sample 16 frames
        indices = sample_frame_indices(clip_len=clip_len, 
                                    current_start=1, 
                                    seg_len=total_frames)
        
        video = read_video_pyav(container, indices)

        inputs = image_processor(list(video), return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            print("output:", outputs.shape)
            
        current_start += step_size
        
 
 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)

# Select the feature type
feature_type = 'i3d'
args = OmegaConf.load(build_cfg_path(feature_type))




args.flow_type = '' # or 'pwc'
args.step_size = 1                # interval, 1
args.stack_size = 16   #15               # window size

args.streams = 'rgb'




from glob import glob
from tqdm import tqdm
import sys, os
sys.path.append('/bask/projects/j/jiaoj-3d-vision/Hao/360x/360x_Video_Experiments')
from db_utils import data_reader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

dr = data_reader()


# This gives one feature vector per 4/25 seconds.
extractor = ExtractI3D(args)



# Extract features
@torch.no_grad()
def get_VideoFeature(files, force=False):

    for file in tqdm(files):
        mp4name = file.split('/')[-1]
        feat_file = file.replace(mp4name, 'video_feat-ours.npy')


        if os.path.exists(feat_file) and not force:
            # print("=== Already exist:", feat_file)
            continue

        try:
            npy = dr.get_framenpy(file)
        except:
            print("=== Error in reading:", file)
            continue
        try:
            with autocast(enabled=True):
                feature_dict = extractor.extract(npy)
        except:
            print("=== Error in extracting:", file)
            continue


        feat = feature_dict['rgb']
        if len(feat.shape) == 3:
            feat = feat.mean(axis=-1)

        print("feat shape:", feat.shape)

        np.save(feat_file, feat)







