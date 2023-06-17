
import os
from glob import glob
import pandas as pd
import numpy as np
import random

cut_number = 6




root = '../Data'

folder = glob(os.path.join(root, "Inside", "*")) + \
          glob(os.path.join(root, "Outside", "*"))

class_folder = glob(os.path.join(root, "Inside", "*")) + \
               glob(os.path.join(root, "Outside", "*"))


def prepare_class_dict(class_folder):
    class2num, num2class = {}, {}
    id = 0
    for each_class in class_folder:
        class_name = each_class.split("/")[-1]
        if class_name not in class2num:
            num2class[id] = class_name
            class2num[class_name] = id
            id += 1

    return class2num, num2class


class2num, num2class = prepare_class_dict(class_folder)
print("class2num: ", class2num)


include_360_cut_video = True
include_360_cut_audio = True

include_front_cut_video = True
include_front_cut_audio = True

include_Clip1_cut_video = True
include_Clip1_cut_audio = True




os.makedirs('prepare_data/360x/csv', exist_ok=True)

contain_video_list_train = pd.DataFrame()
contain_video_list_all = pd.DataFrame()
contain_video_list_test = pd.DataFrame()
classlist = pd.DataFrame()
classlist['class2num'] = class2num
classlist['num2class'] = num2class


print("\nbefore prepareing has: {} videos".format(len(class_folder)))

id = 0

# CLS  ....
for cls in folder:
    cls_folder = glob(os.path.join(cls, "*"))
    print("cls:", cls.split('/')[-1] , " has: {} videos".format(len(cls_folder)))

    random.shuffle(cls_folder)
    val_id = id + 0.85 * len(cls_folder)

    for videoid in cls_folder:

        _360cut_frames_list = glob(os.path.join(videoid, "360/360_panoramic_cut/frames/*"))
        _360cut_audios_list = glob(os.path.join(videoid, "360/360_panoramic_cut/audios/*"))

        _frontcut_frames_list = glob(os.path.join(videoid, "360/front_view_cut/frames/*"))
        _frontcut_audios_list = glob(os.path.join(videoid, "360/front_view_cut/audios/*"))

        _Clip_frames_list = glob(os.path.join(videoid, "Snapchat/clip1/binocular_cut/frames/*"))
        _Clip_audios_list = glob(os.path.join(videoid, "Snapchat/clip1/binocular_cut/audios/*"))

        if include_front_cut_video and len(_frontcut_frames_list) < cut_number:
            print(f"video: {videoid} failed with front cut video")
            continue

        if include_front_cut_audio and len(_frontcut_audios_list) < cut_number:
            print(f"video: {videoid} failed with front cut audio")
            continue

        if include_360_cut_video and len(_360cut_frames_list) < cut_number:
            print(f"video: {videoid} failed with 360 cut video")
            continue

        if include_360_cut_audio and len(_360cut_audios_list) < cut_number:
            print(f"video: {videoid} failed with 360 cut audio")
            continue

        if include_Clip1_cut_audio and len(_Clip_audios_list) < 2:
            print(f"video: {videoid} failed with Clip1 cut audio")
            continue

        if include_Clip1_cut_video and len(_Clip_frames_list) < 2:
            print(f"video: {videoid} failed with Clip1 cut video")
            continue

        class_name = videoid.split("/")[-2]
        InOrOut = videoid.split("/")[-3]

        info = {"id": id, "class": class_name, "classnum": class2num[class_name],
                "InOrOut": InOrOut, "path": videoid}


        contain_video_list_all = contain_video_list_all.append(info, ignore_index=True)
        
        id += 1
        
        if id < val_id:
            contain_video_list_train = contain_video_list_train.append(info, ignore_index=True)
        else:
            contain_video_list_test = contain_video_list_test.append(info, ignore_index=True)
            


print("=========")
print(f"Totally train: {len(contain_video_list_train)} videos")
print(f"Totally test: {len(contain_video_list_test)} videos")
print(f"Totally all: {len(contain_video_list_all)} videos")
print("Done...")


contain_video_list_all.to_csv('prepare_data/360x/csv/all_files.csv', index=False)


print("Test:", contain_video_list_test)
pd.set_option('max_colwidth',200)
print("Test:", contain_video_list_test['path'])

