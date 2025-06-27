import argparse
import os
import shutil
import subprocess, cv2

# conda activate video
# conda install -c conda-forge ffmpeg -y
# conda install -c conda-forge x264 -y
# conda install -c conda-forge opencv -y

def process_folders(root_path):
    # Iterate through each folder in the root directory

    DEBUG = False

    folder_list = os.listdir(root_path)
    folder_list = folder_list[:2] if DEBUG else folder_list  # Limit to first 2 folders for debugging

    print("folder_list = ", folder_list)

    for folder_name in folder_list:
        folder_path = os.path.join(root_path, folder_name)

        if os.path.isdir(folder_path):
            subfolder_snapchat = os.path.join(folder_path, 'Snapchat')

            if os.path.isdir(subfolder_snapchat):

                for f in os.listdir(subfolder_snapchat):
                    file_path = os.path.join(subfolder_snapchat, f)
                    if os.path.isfile(file_path) and f.startswith("._"):
                        os.remove(file_path)

                files = [
                    f for f in os.listdir(subfolder_snapchat)
                    if os.path.isfile(os.path.join(subfolder_snapchat, f)) and f.lower().endswith('.mp4')
                ]


                # Sort files by modification time (oldest to newest)
                files.sort(key=lambda f: os.path.getmtime(os.path.join(subfolder_snapchat, f)))

                for idx, f in enumerate(files, start=1):
                    clip_folder = os.path.join(subfolder_snapchat, f'clip{idx}')
                    os.makedirs(clip_folder, exist_ok=True)

                    src_file = os.path.join(subfolder_snapchat, f)
                    dest_file = os.path.join(clip_folder,      'raw_stereo.mp4')
                    dest_file = dest_file.replace("2024_ready_copy", "2024_ready")

                    print('Copying {} to {}'.format(src_file, dest_file))

                    shutil.copy(src_file, dest_file)
                    # shutil.move(src_file, dest_file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Snapchat subfolders and organize files.')
    parser.add_argument('--root_path', type=str, help='Root directory containing folders to process')
    args = parser.parse_args()

    process_folders(args.root_path)
