
import os
from glob import glob
import pandas as pd


root = '/bask/projects/j/jiaoj-3d-vision/Hao/360x/360data/ActivityNet'

video_list = glob(os.path.join(root, "video", "*"))
audio_list = glob(os.path.join(root, "audio", "*"))

audio_feat = glob(os.path.join(root, "audio_feat", "*"))
video_feat = glob(os.path.join(root, "video_feat", "*"))


video_list = [i.split("/")[-1].rstrip(".mp4") for i in video_list]
audio_list = [i.split("/")[-1].rstrip(".wav") for i in audio_list]

audio_feat = [i.split("/")[-1].rstrip(".npy") for i in audio_feat]
video_feat = [i.split("/")[-1].rstrip(".npy") for i in video_feat]


print("video_list:", video_list)


contain_video_list_all = pd.DataFrame()


for i in video_list:
    if i in video_list and i in audio_list and i in audio_feat and i in video_feat:

        info = {"video_id": i}

        contain_video_list_all = contain_video_list_all.append(info, ignore_index=True)

print(f"Totally all: {len(contain_video_list_all)} videos")
print("Done...")


contain_video_list_all.to_csv('../../data/anet.csv', index=False)




