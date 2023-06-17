


import json

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import shutil  


# root = '../360x/360data/360x_feat/annotation_source'
# json_list = glob(os.path.join(root, "*/*/*/*/*"))

root = '/bask/projects/j/jiaoj-3d-vision/yuqi/Annotation/Inside'
json_list = glob(os.path.join(root, "*/*/*/*"))


target_root = '../360x/360data/360x_feat/annotation/360_source'
# ide/University/04_Bham_GreatHall_20221206_160737/360/*


class2num, num2class = {}, {}
id = 0



            
    
standize = False
for json in tqdm(json_list):
    if standize and json.endswith(".json"):
        target = os.path.join(target_root, json.split('/')[-4]+'_'+json.split('/')[-3]+'.json'  )
        print("Move to:", target)
        shutil.move(json, target)


standize = False
target_root_list = glob(os.path.join(target_root, "*"))
for json in tqdm(target_root_list):
    if standize and json.endswith("json"):
        target = json.replace("json", ".json")


        print("Move to:", target)
        shutil.move(json, target)



json_root = '../360x/360data/360x_feat/annotation/360_source'
json_target_root = '../360x/360data/360x_feat/annotation/360'



print("Strat handling json.....")


from glob import glob
from tqdm import tqdm
import cv2
import json

video_paths = glob('../360x/360data/360x_feat/video/360/*')


split = True

dict_db = {"version": "360x Version 1.0"}
database_db = {}



contain_video_list_all = pd.DataFrame()


for video_path in tqdm(video_paths):


    video_dict = {}

    vid = cv2.VideoCapture(video_path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    video_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps< 1:
        print("fps < 1, skip:", video_path)
        continue

    video_len = int(video_frames / fps)

    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    mp4name = video_path.split("/")[-1].rstrip(".mp4")


    start = int(mp4name.split('_')[-3].rstrip('s'))
    end = int(mp4name.split('_')[-1].rstrip('s'))

    # print(f"video_len: {video_len}s, start {start},  mp4name: {mp4name}")

    jsonname = mp4name.replace(f"_{start}s_to_{end}s", "") + ".json"


    jsonname = os.path.join(json_root, jsonname)

    try:
        with open(jsonname) as f:
            jsonresult = json.load(f)
    except:
        print("json not found:", jsonname)
        continue

    end = start + video_len

    database_db[mp4name] = \
        {"duration": video_len,
         "resolution": [height, width],
         "subset": "training",
         "annotations": [
         ]}


    for k in jsonresult['metadata'].keys():

        clip_duration = jsonresult['metadata'][k]['z']
        clip_label = jsonresult['metadata'][k]['av']['1']
        try:
            c_start = clip_duration[0]
            c_end = clip_duration[1]
        except:
            print("failed:", video_path)
            continue

        if c_start < end and c_end > start:
            final_start = max(c_start, start)
            final_end = min(c_end, end)

            print(clip_label, " : ", final_start, final_end)

            
            
            if clip_label not in class2num:
                class2num[clip_label] = id
                num2class[id] = clip_label
                id += 1


            label_id = class2num[clip_label]
            print("label id:", label_id)
            
            database_db[mp4name]["annotations"].append(
                {"segment": [final_start, final_end], "label": clip_label, "label_id": label_id})


    print(database_db[mp4name]["annotations"])

    info = {"video_id": video_path}

    contain_video_list_all = contain_video_list_all.append(info, ignore_index=True)

labels_dict = pd.DataFrame()

labels_dict['class2num'] = class2num
labels_dict['num2class'] = num2class

dict_db["database"] = database_db

with open(os.path.join(json_target_root, "360panoramic.json"), 'w') as f:
    json.dump(dict_db, f)


contain_video_list_all.to_csv('../../data/360x.csv', index=False)
labels_dict.to_csv('../../data/360x_labeldict.csv', index=False)

print(f"Totally all: {len(contain_video_list_all)} videos")
print("Done...")





