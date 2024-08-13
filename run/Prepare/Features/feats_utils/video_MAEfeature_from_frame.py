from .models.extract_i3d import ExtractI3D
from omegaconf import OmegaConf
import torch
import os, time
from pathlib import Path
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from transformers import VideoMAEConfig, VideoMAEModel, VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from transformers import AutoImageProcessor


# https://github.com/v-iashin/video_features

# conda activate py37
def build_cfg_path(feature_type: str) -> os.PathLike:
    path_base = Path('./feats_utils/configs')
    path = path_base / f'{feature_type}.yml'
    return path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)

# Select the feature type
feature_type = 'i3d'
args = OmegaConf.load(build_cfg_path(feature_type))



model_path = '/bask/projects/j/jiaoj-3d-vision/Hao/360x/360x_Video_Experiments/prepare/prepare_feature/huggingface_cache/models--MCG-NJU--videomae-base-finetuned-kinetics/snapshots/fd26a6d629867cba52f75b5efb6f89993045d3b7'

image_processor = AutoImageProcessor.from_pretrained(model_path)  # -finetuned-kinetics
extractor = VideoMAEModel.from_pretrained(model_path)



args.flow_type = '' # or 'pwc'
step_size = 2                # interval, 1
stack_size = 16   #15               # window size
sample_rate = 1

args.streams = 'rgb'
npyname = f'video_feat-MAE_step{step_size}_size{stack_size}_sample{sample_rate}.npy'

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def get_framefolder(mp4file):
    name = mp4file.split("/")[-1]
    return mp4file.replace(name, "frames")

def get_framenpy(mp4file, size=256, mmap_mode='r'):
    npy = get_framefolder(mp4file) + f'-{size}.npy'
    if not os.path.exists(npy):
        print("=== npy not exists:", npy)
        return None

    try:
        mem = np.load(npy, mmap_mode=mmap_mode)
    except:
        print("=== npy load fails:", npy)
        mem = None

    return mem


# Extract features
@torch.no_grad()
def get_VideoFeature(files, force=False, pretrain_path=None):
    
    if pretrain_path:
        state_dict = torch.load(pretrain_path)
        extractor.name2module['model'].load_state_dict(state_dict)
        
    for file in tqdm(files):
        mp4name = file.split('/')[-1]
        feat_file = file.replace(mp4name, npyname)

        if os.path.exists(feat_file) and not force:
            print("=== Already exist:", feat_file)
            continue

        try:
            npy = get_framenpy(file, mmap_mode=None)
            npy = np.asarray(npy, dtype=np.uint8) 
        except:
            print("=== Error in reading:", file)
            continue
        
        current_start = 0
        feat = []
        start_time = time.time()
        while current_start < (npy.shape[0] - stack_size*sample_rate):
            start_idx = current_start
            end_idx = current_start + stack_size*sample_rate
            current_start += step_size
            inputs = image_processor(list(npy[start_idx:end_idx:sample_rate]), return_tensors="pt")  
                
            try:
                with autocast(enabled=True):
                    feat_cache = extractor(**inputs).last_hidden_state
                feat_cache = feat_cache.mean(axis=-1)
                # [1, 1568, 768]
                if current_start % 100 == 0:
                    print("current step :", current_start, 
                          " with totally:", (npy.shape[0] - stack_size*sample_rate), 
                          " with time:", time.time() - start_time)
         
                feat.append(feat_cache)
            except:
                print("=== Error in extracting:", file)
                continue


        # feat = feature_dict['rgb']
        try:
            feat = np.concatenate(feat, axis=0)
            np.save(feat_file, feat)
            print("feat shape:", feat.shape)
            print("save to :", feat_file)
        except:
            print("=== Error in saving:", file)







