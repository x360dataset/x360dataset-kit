from models.i3d.extract_i3d import ExtractI3D
from omegaconf import OmegaConf
import torch
import os
from pathlib import Path

def build_cfg_path(feature_type: str) -> os.PathLike:
    path_base = Path('./configs')
    path = path_base / f'{feature_type}.yml'
    return path




device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)


# Select the feature type
feature_type = 'i3d.py'

# Load and patch the config
args = OmegaConf.load(build_cfg_path(feature_type))
args.flow_type = 'raft' # or 'pwc'

print("args:", args)


extractor = ExtractI3D(args)




video_paths = ['/sample/v_GGSY1Qvo990.mp4']





# Extract features
for video_path in video_paths:
    print(f'Extracting for {video_path}')
    feature_dict = extractor.extract(video_path)
    [(print(k), print(v.shape), print(v)) for k, v in feature_dict.items()]
    # rgb
    # (5, 1024)

    # flow
    # (5, 1024)



