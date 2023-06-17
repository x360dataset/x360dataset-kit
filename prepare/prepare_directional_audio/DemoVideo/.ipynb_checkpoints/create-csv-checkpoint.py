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
        fieldnames = ["path"] # "train/ test/ val  "#list(data_list[0].keys())
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for info in data_list:
            print("info)
            writer.writerow(info)
            
    print('{} items saved to {}.'.format(len(data_list), filepath))



def exam(args):
    # read_path = 'ProcessedData'
    read_path = args.data_path
    split_path = f'./data-split'
    os.makedirs(split_path, exist_ok=True)
    
    data_list = glob.glob(f'{read_path}/*')
    data_list.sort()
    broken_list = []
    sample_rate_list = []
    for item in tqdm(data_list):
        meta_path = os.path.join(item, 'meta.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        sample_rate_list.append(meta_dict['audio_sample_rate'])
        # import pdb; pdb.set_trace() 
        audio_path = os.path.join(item, 'audio', 'audio.wav')
        audio, audio_rate = sf.read(audio_path, dtype='int16')
        cond_1 = meta_dict['frame_num'] <= 98
        cond_2 = not meta_dict['ifstereo']
        cond_3 = np.abs(audio.shape[0] / audio_rate - meta_dict['frame_num'] / meta_dict['frame_rate'] ) > 0.1
        if cond_1 or cond_2 or cond_3:
            broken_list.append(
                {
                    'broken url': meta_dict['u_id'],
                    'broken time': item,
                    'frame num': meta_dict['frame_num'],
                    'audio length': audio.shape[0] / audio_rate
                }
            )
    if len(broken_list) > 0:
        write_csv(broken_list, os.path.join(split_path, 'broken.csv'))
    else:
        tqdm.write('None')
    tqdm.write(f'Sample rate: max = {np.max(sample_rate_list)}, min = {np.min(sample_rate_list)}')



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
        path = video  # os.path.join('./data/DemoVideo', video)
        
        meta_path = os.path.join(video, 'meta.json')
        
        try:  
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)
        except:
            print("Broken meta:", meta_path)
            continue
            
        frame_num = meta_dict['frame_num']
        frame_rate = meta_dict['frame_rate']
        # import pdb; pdb.set_trace()
        audio_path = os.path.join(video, 'audio', 'audio.wav')
        audio, audio_rate = sf.read(audio_path, dtype='int16')
        cond_2 = not meta_dict['ifstereo']
        cond_3 = np.abs(audio.shape[0] / audio_rate - meta_dict['frame_num'] / meta_dict['frame_rate'] ) > 0.1

        if cond_2 or cond_3:
            continue


        if name == 'train':
            sample_list.append({'path': path})
            
        elif name == 'val':
            # start_times = np.random.choice(frame_num - frame_rate * 3, 5, replace=False)
            start_times = np.random.choice(frame_num - frame_rate * 3, 10, replace=False)

            for i in start_times:
                sample = {
                    'path': path,
                    'start_time': i
                }
                sample_list.append(sample)
        elif name == 'test':
            start_times = np.random.choice(frame_num - frame_rate * 3, 8, replace=False)
            for i in start_times:
                sample = {
                    'path': path,
                    'start_time': i
                }
                sample_list.append(sample)
    if not args.unshuffle:
        random.shuffle(sample_list)
    return sample_list



def main(args):
    test_num = 10
    home_address = "/bask/projects/j/jiaoj-3d-vision/360XProject/Data/*/*/*"
    # [Inside, Outside]/ [Type]/ [Video ID]/ [360 Video or 360]
    
    video_root = os.path.join(home_address, "360*")
    
    out_root = video_root 
    
    os.makedirs(out_root, exist_ok=True)
    
    video_list = glob.glob(f'{video_root}')  # Video folder
    print("data_list:", video_list)
    
    video_list.sort()
    
    data_list = video_list[:test_num]
    print("data_list:", data_list)
    
    
    
    begin = 0
    name = 'vis'
    sample_list = create_list_for_video(args, 'train', data_list)
    print("sample_list:", sample_list)
    
    split_path = "/bask/projects/j/jiaoj-3d-vision/360XProject/Data/Meta"
    csv_name = f'{split_path}/{name}.csv'
    
    write_csv(sample_list, csv_name)
        

if __name__ == "__main__":
    args = parser.parse_args()
    # if args.exam:
    #     exam(args)
    # else:
    main(args)