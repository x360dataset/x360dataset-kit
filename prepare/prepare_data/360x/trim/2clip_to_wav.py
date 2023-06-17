import os
from glob import glob

root = '../Data'

# $Root/ (Inside_Outside)/ Location(Label)/ Video_ID


def generate_wav_from_mp4list(_list, force=False):

    for i, item in enumerate(_list):
        if i % 500 == 0:
            print('*******************************************')
            print('{}/{}'.format(i, len(_list)))
            print('*******************************************')

        mp4_filename = item
        mp4name = item.split("/")[-1]

        target_folder = os.path.join(item.rstrip(mp4name), 'audios')
        os.makedirs(target_folder, exist_ok=True)

        if mp4name.endswith(".mp4") and mp4name.startswith("cut_"):
            wav_filename = os.path.join(target_folder,
                                        mp4name.replace(".mp4", ".wav"))

            if os.path.exists(wav_filename) and not force:
                pass
            else:
                os.system('ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

            # Handle if File Empty

    print("Done")




_360_list =  glob(os.path.join(root, "Inside", "*", "*", "360/360_panoramic_cut/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "360/360_panoramic_cut/*"))

_360_front_list = glob(os.path.join(root, "Inside", "*", "*", "360/front_view_cut/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "360/front_view_cut/*"))


_Snapchat_list = glob(os.path.join(root, "Inside", "*", "*", "Snapchat/*/binocular_cut/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "Snapchat/*/binocular_cut/*"))



# print(f"processing {len(_Snapchat_list)} videos...")
generate_wav_from_mp4list(_Snapchat_list)  #  , force=True)


print(f"processing {len(_360_list)} videos...")
generate_wav_from_mp4list(_360_list)

print(f"processing {len(_360_front_list )} videos...")
generate_wav_from_mp4list(_360_front_list)
