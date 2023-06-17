


import os
from glob import glob
from datetime import datetime, timedelta

root = '../Data'
TIME_FROMAT = '%H:%M:%S'


num_of_trim = 6
trim_interval = 10    # seconds

# $Root/ (Inside_Outside)/ Location(Label)/ Video_ID

def seconds_to_time(seconds):
    return str(timedelta(seconds=seconds))


def generate_wav_from_mp4list(_list, videoname, force=False):

    for i, item in enumerate(_list):
        if i % 500 == 0:
            print('*******************************************')
            print('{}/{}'.format(i, len(_list)))
            print('*******************************************')

        mp4_filename = item
        mp4name = item.split("/")[-1]

        if mp4name == videoname:    # filter
            outfolder = item.rstrip(".mp4") + "_cut"
            os.makedirs(outfolder, exist_ok=True)

            for trim in range(num_of_trim):

                start = seconds_to_time(trim * trim_interval)
                end = seconds_to_time((trim +1) * trim_interval)
                trim_filename = os.path.join(outfolder, f"cut_{trim}_start_{trim * trim_interval}"
                                                        f"_end_{(trim +1) * trim_interval}.mp4")
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

print(f"processing {len(_Snapchat_list)} videos...")

generate_wav_from_mp4list(_Snapchat_list, "binocular.mp4") #, force=True)









