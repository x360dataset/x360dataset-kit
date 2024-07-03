import numpy as np

def get_audio_feat(mp4file):
    name = mp4file.split("/")[-1]
    return np.load(mp4file.replace(name, "audio_feat.npy"))

def get_video_feat(mp4file):
    name = mp4file.split("/")[-1]
    return np.load(mp4file.replace(name, "video_feat.npy"))

