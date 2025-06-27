import argparse
import os
import shutil

def process_folders(root_path):
    # Iterate through each folder in the root directory
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        if os.path.isdir(folder_path):
            subfolder_360 = os.path.join(folder_path, '360')

            if os.path.isdir(subfolder_360):

                for f in os.listdir(subfolder_360):
                    file_path = os.path.join(subfolder_360, f)
                    if os.path.isfile(file_path) and f.startswith("._"):
                        os.remove(file_path)

                files = [f for f in os.listdir(subfolder_360)
                         if os.path.isfile(os.path.join(subfolder_360, f))]

                if len(files) != 2:
                    print(f"Skipping {subfolder_360}: Expected exactly 2 files, found {len(files)}")
                    continue

                file_paths = [os.path.join(subfolder_360, f) for f in files]

                size_0 = os.path.getsize(file_paths[0])
                size_1 = os.path.getsize(file_paths[1])

                if size_0 >= size_1:
                    panoramic_file = file_paths[0]
                    front_file = file_paths[1]
                else:
                    panoramic_file = file_paths[1]
                    front_file = file_paths[0]

                panoramic_dest = os.path.join(subfolder_360,     '360_panoramic.mp4')
                front_dest     = os.path.join(subfolder_360,     'front_view.mp4')

                print(f"Renaming {panoramic_file} to {panoramic_dest}")
                print(f"Renaming {front_file} to {front_dest}")

                shutil.move(panoramic_file, panoramic_dest)
                shutil.move(front_file, front_dest)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process 360 subfolders and rename files.')
    parser.add_argument('--root_path', type=str, help='Root directory containing folders to process')
    args = parser.parse_args()

    process_folders(args.root_path)
