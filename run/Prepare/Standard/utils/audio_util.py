import os
from glob import glob


def generate_wav_from_mp4list(_list, force=False, DEBUG=False, verbose=False):

    for i, item in enumerate(_list):
        if i % 200 == 0:
            print('*******************************************')
            print('{}/{}'.format(i, len(_list)))
            print('*******************************************')

        source = item
        mp4name = item.split("/")[-1]

        # os.makedirs(target_folder, exist_ok=True)

        if mp4name.endswith(".mp4"):
            wav_filename = source.replace(".mp4", ".wav")

            if os.path.exists(wav_filename) and not force:
                if verbose:
                    print("Already exists: ", wav_filename)
                continue
            else:
                try:
                    os.system('ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 {} -loglevel error'.format(
                        source, wav_filename))
                except:
                    print("=== Audio Error: ", source)
                    continue

            if verbose:
                print("source: ", source, "\n  -->target_filename: ", wav_filename)

    print("Done")

