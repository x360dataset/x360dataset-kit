import pdb
import subprocess
import argparse
import re
import cv2
import sys
import os
import glob
import json
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm
import soundfile as sf
import shutil

parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--split', type=int, default=0, help='i split of videos to process')
parser.add_argument('--total', type=int, default=1, help='total splits')
parser.add_argument('--frame_rate', type=int, default=10, help='total splits')
parser.add_argument('--dataset_name', default='', type=str)
parser.add_argument('--input_file', default='', type=str)


def get_frame(video_path, save_path, fps=10, resol=480):
    
    # -y -i yes rebuild...
    try:
        command = f'ffmpeg -v quiet  -i \"{video_path}\" -f image2 -vf \"scale=-1:{resol},fps={fps}\" -qscale:v 3 \"{save_path}\"/frame%04d.jpg'
        os.system(command)
    except:
        print("frame all ready export")
        
    frame_info = {
        'frame_num': len(os.listdir(save_path)),
        'frame_rate': fps
    }
    return frame_info


def get_audio(video_path, save_path):
    
    audio_path = os.path.join(save_path, 'audio.wav')
    if not os.path.exists(audio_path):
        command = f"ffmpeg -v quiet -y -i \"{video_path}\" \"{audio_path}\""
        os.system(command)

    try:
        audio, audio_rate = sf.read(audio_path, start=0, stop=10000, dtype='int16')
    except (RuntimeError, TypeError, NameError):
        return None

    # audio, audio_rate = sf.read(audio_path, start=0, stop=10000, dtype='int16')
    ifstereo = (len(audio.shape) == 2)
    audio_info = {
        'audio_sample_rate': audio_rate,
        'ifstereo': ifstereo
    }
    return audio_info


def get_meta(clip, json_path, frame_info, audio_info):
    video_info = {}

    meta_dict = {**video_info, **frame_info, **audio_info}
    with open(json_path, 'w') as fp:
        json.dump(meta_dict, fp, sort_keys=False, indent=4)



def main(focus=False):
    import cv2
    args = parser.parse_args()

    root = "../360XProject/Data"
    # Change to Cut? and the save issue..

    video_list = glob.glob(os.path.join(root, "*", "*", "*", "360/front_view_cut/*.mp4"))   # 360_panoramic_cut
    print(f"processing {len(video_list)} videos...")

    video_list.sort()

    vidcap = cv2.VideoCapture(video_list[50])
    # fps = vidcap.get(cv2.CAP_PROP_FPS)#x/
    # print("frame rate is ", fps)

    for video in tqdm(video_list, desc=f'Video Processing ID = {str(args.split).zfill(2)}'):
        video_name = video.split('/')[-1][:-4]  # remove .mp4

        if not video.endswith('.mp4'): # or not video_name.startswith("cut_"):
            continue
        
        video_name = video.split('/')[-1].rstrip('.mp4')

        processed_path = video.rstrip(video.split('/')[-1]).rstrip('/')

        print("processed_path: ", processed_path)
        print("process name:", video_name)
        
        # os.system(f"rm -rf {os.path.join(processed_path, 'at')}")
        
        frame_path = os.path.join(processed_path, 'at', video_name, 'frames')
        audio_path = os.path.join(processed_path, 'at', video_name, 'audio')
        meta_path = os.path.join(processed_path, 'at', video_name, 'meta')
        
        if os.path.exists(frame_path):
            print(processed_path, " is exists")
            
            continue
        
        os.makedirs(meta_path, exist_ok=True)


        # if not os.path.exists(frame_path):
        print("Get Frames...")  # About 3mins for a 5mins 360 video
        os.makedirs(frame_path, exist_ok=True)
        frame_info = get_frame(video, frame_path, args.frame_rate) # fps

        # audio
        # if not os.path.exists(audio_path):
        print("Get Audios...")
        os.makedirs(audio_path, exist_ok=True)
        audio_info = get_audio(video, audio_path)

        if audio_info is None:
            tqdm.write(f'{processed_path} is broken')
            shutil.rmtree(processed_path)
            continue


        # meta data
        get_meta(video, os.path.join(meta_path, video_name + "_meta.json"),
                 frame_info, audio_info)

        tqdm.write(f'Finished: {video_name}!')


if __name__ == "__main__":
    main()
