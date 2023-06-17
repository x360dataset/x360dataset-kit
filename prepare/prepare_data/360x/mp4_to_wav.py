import os
from glob import glob

root = '../Data'

# $Root/ (Inside_Outside)/ Location(Label)/ Video_ID


def generate_wav_from_mp4list(_list, videoname, force=False):

    for i, item in enumerate(_list):
        if i % 500 == 0:
            print('*******************************************')
            print('{}/{}'.format(i, len(_list)))
            print('*******************************************')

        mp4_filename = item
        mp4name = item.split("/")[-1]
        if mp4name == videoname:
            wav_filename = item.replace(".mp4", ".wav")

            if os.path.exists(wav_filename) and not force:
                pass
            else:
                os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

    print("Done")




_360_list =  glob(os.path.join(root, "Inside", "*", "*", "360/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "360/*"))


_Snapchat_list = glob(os.path.join(root, "Inside", "*", "*", "Snapchat/*/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "Snapchat/*/*"))




print(f"processing {len(_360_list)} videos...")
generate_wav_from_mp4list(_360_list, "360_panoramic.mp4")

print(f"processing {len(_360_list)} front view videos...")
generate_wav_from_mp4list(_360_list, "front_view.mp4")

print(f"processing {len(_Snapchat_list)} videos...")
generate_wav_from_mp4list(_Snapchat_list, "binocular.mp4", force=True)
