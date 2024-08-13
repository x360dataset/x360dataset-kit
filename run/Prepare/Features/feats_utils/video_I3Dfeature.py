from .models.extract_i3d import ExtractI3D
from omegaconf import OmegaConf
import torch
import os
from pathlib import Path
import numpy as np
import git, sys
# get_top_level_directory
repo = git.Repo(".", search_parent_directories=True)
top_level_directory = repo.working_tree_dir
sys.path.append(top_level_directory)
from lib.database import database
from torch.cuda.amp import autocast, GradScaler
# ref https://github.com/v-iashin/video_features


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


args.flow_type = ''  # or 'pwc'
args.step_size = 1   # interval, 1
args.stack_size = 16
args.streams = 'rgb'

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


# This gives one feature vector per 4/25 seconds.
extractor = ExtractI3D(args)


def get_framefolder(mp4file):
    name = mp4file.split("/")[-1]
    return mp4file.replace(name, "frames")

def get_framenpy(mp4file, size=256, mmap_mode='r'):
    npy = get_framefolder(mp4file) + f'-{size}.npy'
    if not os.path.exists(npy):
        print("=== frame npy not exists:", npy)
        return None

    try:
        mem = np.load(npy, mmap_mode=mmap_mode)
    except:
        print("=== frame npy load fails:", npy)
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
        feat_file = file.replace(mp4name, 'video_feat.npy')

        if os.path.exists(feat_file) and not force:
            continue

        npy = get_framenpy(file)

        try:
            npy = get_framenpy(file)
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








