from models.i3d.extract_i3d import ExtractI3D
from omegaconf import OmegaConf
import torch
import os
from pathlib import Path
import numpy as np

# https://github.com/v-iashin/video_features

# conda activate py37
def build_cfg_path(feature_type: str) -> os.PathLike:
    path_base = Path('./video_features/configs')
    path = path_base / f'{feature_type}.yml'
    return path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)

# Select the feature type
feature_type = 'i3d'
args = OmegaConf.load(build_cfg_path(feature_type))
args.flow_type = 'raft' # or 'pwc'
args.step_size = 16  # interval
args.stack_size = 64 # window size

args.streams = 'rgb'



# This gives one feature vector per 16/25 = 0.64 seconds.
extractor = ExtractI3D(args)






from glob import glob
from tqdm import tqdm

video_paths = glob('/bask/projects/j/jiaoj-3d-vision/Hao/360x/360data/360x_feat/360_panoramic/*')
target = '/bask/projects/j/jiaoj-3d-vision/Hao/360x/360data/360x_feat/video_feat/360_panoramic'

os.makedirs(target, exist_ok=True)

from random import shuffle
shuffle(video_paths)

# Extract features
for video_path in tqdm(video_paths):

    output = os.path.join(target, video_path.split('/')[-1].rstrip('.mp4') + '.npy')

    if os.path.exists(output):
        print(f'Passing for {video_path}')
        continue

    with torch.no_grad():
        feature_dict = extractor.extract(video_path)
    feat = feature_dict['rgb']


    # print("feat.shape:", feat.shape)
    np.save(output, feat)





