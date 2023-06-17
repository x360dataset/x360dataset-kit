import os
import numpy as np
import glob
import argparse
import random
import json
from tqdm import tqdm
import csv
import cv2
import soundfile as sf

# python create-csv.py --dataset_name='Youtube-RacingCar' --type='vis_res' --data_split='1:0:0'
# python create-csv.py --dataset_name='Youtube-Inthewild' --type='' --data_split='1:0:0' 


parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--exam', default=False, action='store_true')
parser.add_argument('--type', default='', type=str)
parser.add_argument('--dataset_name', default='', type=str)
parser.add_argument('--data_split', default='8:1:1', type=str)
parser.add_argument('--split_by_video', default=False, action='store_true')
parser.add_argument('--unshuffle', default=False, action='store_true')

random.seed(1234)


def write_csv(data_list, filepath):
    # import pdb; pdb.set_trace()
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ["path"]  # "train/ test/ val  "#list(data_list[0].keys())
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for info in data_list:
            # print("info: ", info)
            writer.writerow(info)

            print('{} items saved to {}.'.format(len(data_list), filepath))


def create_list_for_video(args, name, video_list):
    # import pdb; pdb.set_trace()

    sample_list = []
    if args.split_by_video:
        clip_list = []
        for video in video_list:
            temp = glob.glob(f'{video}/*')
            temp.sort()
            clip_list += temp
        new_video_list = clip_list
    else:
        new_video_list = video_list

    for video in tqdm(new_video_list):
        path = video
        video_name = path.split('/')[-1]

        if not video.endswith('.mp4'): # or not video_name.startswith("cut_"):
            continue

        video = video.rstrip(video_name).rstrip("/")
        
        meta_path = os.path.join(video, 'at',video_name.rstrip(".mp4"), 
                                 "meta", video_name.rstrip(".mp4")+'_meta.json')

        try:
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)
        except:
            print("Broken meta:", meta_path)
            continue

        frame_num = meta_dict['frame_num']
        frame_rate = meta_dict['frame_rate']
        
        print("frame_rate :", frame_rate)

        # import pdb; pdb.set_trace()

        audio_path = os.path.join(video, 'at', video_name.rstrip(".mp4"), 'audio', 'audio.wav')
        audio, audio_rate = sf.read(audio_path, dtype='int16')

        cond_2 = not meta_dict['ifstereo']


        if frame_rate < 1:
            cond_3 = True
        else:
            cond_3 = np.abs(audio.shape[0] / audio_rate - meta_dict['frame_num'] / frame_rate) > 0.1

        if cond_2 or cond_3:
            continue

        if name == 'train':
            sample_list.append({'path': path})

    if not args.unshuffle:
        random.shuffle(sample_list)

    return sample_list




def main(args):
    test_num = -1

    root = "../360XProject/Data"
    video_list = glob.glob(os.path.join(root, "*", "*", "*", "360/front_view_cut/*.mp4"))
    print(f"processing {len(video_list)} videos...")


    # _Snapchat_list = glob.glob(os.path.join(root, "*", "*", "*", "Snapchat/*/binocular_cut/*"))
    # video_list += _Snapchat_list
    
    video_list.sort()

    data_list = video_list[:test_num]

    data_list = [i for i in data_list if "*" not in i]

    begin = 0
    name = 'Meta'
    sample_list = create_list_for_video(args, 'train', data_list)
    print("sample_list length:", len(sample_list))

    split_path = "../360XProject/Data"
    csv_name = f'{split_path}/{name}.csv'

    write_csv(sample_list, csv_name)


if __name__ == "__main__":
    args = parser.parse_args()
    
    # if args.exam:
    #     exam(args)
    # else:
    
    main(args)