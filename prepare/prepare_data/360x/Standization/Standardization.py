import os
from glob import glob
import shutil
import collections
import cv2
import numpy as np


root = "../Data"


def getFileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)


# ============== Rename ==============
# $Root/ (Inside_Ouside)/ Location(Label)/ Video_ID

VideosID_list = glob(os.path.join(root, "Inside", "*", "*")) + \
                glob(os.path.join(root, "Outside", "*", "*"))  # VideoID



# Rename
process_insta = False
process_snapchat = True

# First 10s seconds of the video

print(f"processing {len(VideosID_list)} videos...")



for i, item in enumerate(VideosID_list):
    if i % 5 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(VideosID_list)))
        print('*******************************************')

    _360_folder = glob(os.path.join(item, "360*/*"))
    _clip_folder = glob(os.path.join(item, "Snapchat*/*"))



    ##### Handling Insta Data
    if process_insta:
        for file in _360_folder:
            if file.lower().endswith(".mp4"):
                print(f" Size: {getFileSize(file)} MB of {file}")

                name = file.split("/")[-1]
                MB = getFileSize(file)

                if MB > 2000:
                    shutil.move(file, file.replace(name, "360_panoramic.mp4"))

                else:
                    shutil.move(file, file.replace(name, "front_view.mp4"))


    if process_snapchat:
        ##### Handling Snapchat Data
        # front_view.mp4  stereo_view.mp4
        for clip in _clip_folder:
            SnapChat_list = {}  # name: MB
            _file_folder = glob(os.path.join(clip, "*"))
            w_list = []
            h_list = []
            name_list = []

            for file in _file_folder:

                if file.lower().endswith(".mp4"):

                    name = file.split("/")[-1]  # Use MB
                    vcap = cv2.VideoCapture(file)  # 0=camera

                    if vcap.isOpened():
                        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
                        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
                        h_list.append(height)
                        w_list.append(width)
                        name_list.append(name)


            if len(h_list) == 3:
                h_list = np.array(h_list)
                w_list = np.array(w_list)
                sort = w_list.argsort()[::-1]
                print("sort: ", sort)

                SnapChat_list[name_list[sort[0]]] = 'binocular'

                # bigger to small
                SnapChat_list[name_list[sort[1]]] = 'monocular'
                SnapChat_list[name_list[sort[2]]] = 'cropped'


            if len(SnapChat_list.keys()) > 3:
                print("TODO!!!!")

            if len(SnapChat_list.keys()) == 3:

                # SnapChat_list = sorted(SnapChat_list.items(),
                #                        key=lambda x: x[1],
                #                        reverse=True)
                if process_snapchat:
                    for k in SnapChat_list.keys():
                        print(k, " ,", SnapChat_list[k])

                        if SnapChat_list[k] == 'binocular':
                            # print(k, ": ", "binocular")
                            shutil.move(os.path.join(clip, k),
                                        os.path.join(clip, "binocular_rename.mp4"))

                        if SnapChat_list[k] == 'monocular':
                            # print(k, ": ", "monocular")
                            shutil.move(os.path.join(clip, k),
                                        os.path.join(clip, "monocular_rename.mp4"))

                        if SnapChat_list[k] == 'cropped':
                            # print(k, ": ", "cropped")
                            shutil.move(os.path.join(clip, k),
                                        os.path.join(clip, "cropped_rename.mp4"))

                if process_snapchat:

                    shutil.move(os.path.join(clip, "monocular_rename.mp4"),
                                os.path.join(clip, "monocular.mp4")
                                )

                    shutil.move(os.path.join(clip, "binocular_rename.mp4"),
                                os.path.join(clip, "binocular.mp4")
                                )

                    shutil.move(os.path.join(clip, "cropped_rename.mp4"),
                                os.path.join(clip, "cropped.mp4")
                                )

            print(SnapChat_list)

    # binocular
    # monocular
    # cropped



    # mp4_filename = item
    # wav_filename = item.replace("video.mp4", 'audio.wav')
    #
    # if os.path.exists(wav_filename):
    #     pass
    # else:
    #     os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))


print("Done")
