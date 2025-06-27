import os, cv2
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np

def seconds_to_time(seconds):
    return str(timedelta(seconds=seconds))

# Original:
# 360/360_panoramic.mp4
# stereo/binocular/clip*/binocular.mp4

# Target:
# 360_trim/360_panoramic_trim/cut0_0s_to_10s/cut0_0s_to_10s.mp4
# stereo_trim/binocular_trim/clip*/cut0_0s_to_10s/cut0_0s_to_10s.mp4


original_datafolder = "/bask/projects/j/jiaoj-3d-vision/360XProject/x360dataset"
trim_datafolder = "/bask/projects/j/jiaoj-3d-vision/360XProject/Extractx360dataset"



def get_outfolder(source, videoname):
    source = source.replace(original_datafolder, trim_datafolder)

    suffix = "_trim"
    is_snapchat = False

    if "clip" in source and "stereo" in source:
       is_snapchat = True

    if not is_snapchat:
        print("not snapchat:", source, videoname)
        folder = source.rstrip(videoname).rstrip("/") + suffix  # 360
        outfolder = os.path.join(folder, videoname.rstrip(".mp4") + suffix) # 360_trim/360_panoramic_trim


    else:
        folder = source.rstrip(videoname).rstrip("/") # binocular or monocular
        clip = folder.split("/")[-1]

        folder = folder.rstrip(clip).rstrip("/") + suffix       # Snapchat_trim
        outfolder = os.path.join(folder, videoname.rstrip(".mp4") + suffix, clip)

    os.makedirs(outfolder, exist_ok=True)
    return outfolder


def trim_video_from_mp4list(_list, videoname,
                            force=False, DEBUG=False, verbose=False,
                            num_of_trim=6, trim_interval=10,
                            is_getall=False):
    # Trim first 60 seconds into 6 clips, each 10 seconds
    PROCESSED = 0

    for i, item in tqdm(enumerate(_list)):
        if i % 500 == 0:
            print('*******************************************')
            print('{}/{}'.format(i, len(_list)))
            print('*******************************************')

        source = item
        mp4name = item.split("/")[-1]

        if videoname in mp4name:    # filter
            outfolder = get_outfolder(source, mp4name)

            video = cv2.VideoCapture(source)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_number = video.get(cv2.CAP_PROP_FRAME_COUNT)

            try:
                duration = frame_number / fps
            except:
                print("=== Video Error: ", source)
                print("=== frame_number: ", frame_number)
                print("=== fps: ", fps)
                continue

            num_of_trim__ = int(np.ceil(duration / trim_interval))

            if is_getall:
                pass
            else:
                num_of_trim_ = np.minimum(num_of_trim, num_of_trim__)

            if num_of_trim_ <= 0:
                print("num_of_trim_ = 0:", mp4name)

            for trim in range(num_of_trim_):

                start = seconds_to_time(trim * trim_interval)
                end = seconds_to_time((trim + 1) * trim_interval)
                cut_name = f"cut_{trim}_start_{trim * trim_interval}_end_{(trim +1) * trim_interval}"

                trim_folder = os.path.join(outfolder, cut_name)
                os.makedirs(trim_folder, exist_ok=True)
                target_filename = os.path.join(trim_folder, cut_name + ".mp4")

                # print("target_filename:", target_filename)
                if (not os.path.exists(target_filename) or force):
                    print("source: ", source, "\n  -->target_filename: ", target_filename)
                    os.system('ffmpeg -y -i {} -ss {} -to {} -c:v copy -c:a copy  {} -loglevel error'.format(
                            source, start, end, target_filename))
                else:
                    if verbose:
                        print("Already exists: ", target_filename)
                PROCESSED += 1

    print("Done... with {} processed".format(PROCESSED))
