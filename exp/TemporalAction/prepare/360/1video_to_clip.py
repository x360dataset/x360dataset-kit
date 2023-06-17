

import cv2
import os
from glob import glob
from datetime import datetime, timedelta

root = '../Data'
TIME_FROMAT = '%H:%M:%S'

target = '../360x/360data/360x_feat'



num_of_trim = 6  #
trim_interval = 10    # seconds

# $Root/ (Inside_Outside)/ Location(Label)/ Video_ID

def seconds_to_time(seconds):
    return str(timedelta(seconds=seconds))


def generate_wav_from_mp4list(_list, videoname, force=False):

    videofolder = videoname.rstrip(".mp4")
    outfolder = os.path.join(target, videofolder)
    os.makedirs(outfolder, exist_ok=True)

    for i, item in enumerate(_list):
        if i % 500 == 0:
            print('*******************************************')
            print('{}/{}'.format(i, len(_list)))
            print('*******************************************')

        if not item.endswith(".mp4"):
            continue

        mp4_filename = item
        # print("mp4_filename:", mp4_filename)


        mp4name = item.split("/")[-4] + '_' + item.split("/")[-3]   #+ '.mp4'

        vid = cv2.VideoCapture(item)

        fps = int(vid.get(cv2.CAP_PROP_FPS))
        video_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

        video_len = int(video_frames / fps)
        print(f"video_len: {video_len} seconds")

        num_of_trim = video_len // trim_interval + 1


        if item.split('/')[-1] == videoname:  # filter

            for trim in range(num_of_trim):

                start = seconds_to_time(trim * trim_interval)
                end = seconds_to_time((trim + 1) * trim_interval)
                trim_filename = os.path.join(outfolder, f"{mp4name}_{trim * trim_interval}s"
                                                        f"_to_{(trim + 1) * trim_interval}s.mp4")

                if os.path.exists(trim_filename) and not force:
                    pass
                else:
                    os.system('ffmpeg -y -i {} -ss {} -to {} -c:v copy -c:a copy  {}'.format(
                        mp4_filename, start, end, trim_filename))

    print("Done")



_360_list =  glob(os.path.join(root, "Inside", "*", "*", "360/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "360/*"))


_Snapchat_list = glob(os.path.join(root, "Inside", "*", "*", "Snapchat/*/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "Snapchat/*/*"))


print(f"processing {len(_360_list)} videos...")

generate_wav_from_mp4list(_360_list, "360_panoramic.mp4")
generate_wav_from_mp4list(_360_list, "front_view.mp4")










