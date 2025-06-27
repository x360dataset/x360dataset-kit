import argparse
import os
import shutil, glob
import subprocess, cv2
from tqdm import tqdm

# conda activate video
# conda install -c conda-forge ffmpeg -y
# conda install -c conda-forge x264 -y
# conda install -c conda-forge opencv -y
# pip install tqdm

def process_clips(root_path):
    DEBUG = False

    folder_list = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    folder_list = folder_list[:2] if DEBUG else folder_list

    for folder_name in tqdm(folder_list):
        snapchat_path = os.path.join(root_path, folder_name, 'Snapchat')

        if not os.path.isdir(snapchat_path):
            continue

        clip_folders = [cf for cf in os.listdir(snapchat_path) if cf.startswith('clip')]
        clip_folders = clip_folders[:2] if DEBUG else clip_folders  # Limit to first 2 folders for debugging


        for clip_folder in clip_folders:
            clip_path = os.path.join(snapchat_path, clip_folder)
            raw_file = os.path.join(clip_path, 'raw_stereo.mp4')

            pattern = os.path.join(clip_path, '._*')

            # Find and remove all matching files
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")


            if not os.path.isfile(raw_file):
                print(f"Skipping {clip_path}, raw_stereo.mp4 not found.")
                continue

            cap = cv2.VideoCapture(raw_file)
            if not cap.isOpened():
                print(f"Failed to open {raw_file}")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if width == 0 or height == 0:
                print(f"Invalid video dimensions in {raw_file}")
                continue

            out_h = height // 2.47  # 2
            out_w = out_h  # Square crop

            y = height // 3.35   # 4
            x_left = y
            x_right = width // 2 + y

            left_cmd = f'[0:v]crop={out_w}:{out_h}:{x_left}:{y}[left]; '
            right_cmd = f'[0:v]crop={out_w}:{out_h}:{x_right}:{y}[right]; '


            processed_file = os.path.join(clip_path, 'stereo.mp4')
            # To stereo.mp4
            cmd = [
                'ffmpeg',
                '-loglevel', 'quiet',
                '-i', raw_file,
                '-filter_complex',
                f'{left_cmd}{right_cmd}[left][right]hstack=inputs=2[v]',
                '-map', '[v]',
                '-c:v', 'libx264',
                '-crf', '18',
                '-y', processed_file
            ]

            if not os.path.isfile(processed_file):
                print(f"Processing {raw_file} -> {processed_file}")
                subprocess.run(cmd, check=True)


            processed_file = os.path.join(clip_path, 'mono.mp4')
            cmd = [
                'ffmpeg',
                '-loglevel', 'quiet',
                '-i', raw_file,
                '-filter_complex',
                f'[0:v]crop={out_w}:{out_h}:{x_left}:{y}[v]',
                '-map', '[v]',
                '-c:v', 'libx264',
                '-crf', '18',
                '-y', processed_file
            ]
            if not os.path.isfile(processed_file):
                print(f"Processing {raw_file} -> {processed_file}")
                subprocess.run(cmd, check=True)


            # To Crop mp4
            radius = out_w / 2

            # Calculate the side of the largest inscribed square inside the circle
            inner_rect_side = int(radius * (2 ** 0.5))

            # Center the inner square inside the circular area
            inner_rect_x = int(x_left + (radius - inner_rect_side / 2))
            inner_rect_y = int(y + (radius - inner_rect_side / 2))

            processed_file = os.path.join(clip_path, 'mono_crop.mp4')

            cmd = [
                'ffmpeg',
                '-loglevel', 'quiet',
                '-i', raw_file,
                '-filter_complex',
                f'[0:v]crop={inner_rect_side}:{inner_rect_side}:{inner_rect_x}:{inner_rect_y}[v]',
                '-map', '[v]',
                '-c:v', 'libx264',
                '-crf', '18',
                '-y', processed_file
            ]
            if not os.path.isfile(processed_file):
                print(f"Processing {raw_file} -> {processed_file}")
                subprocess.run(cmd, check=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Snapchat subfolders and organize files.')
    parser.add_argument('--root_path', type=str, help='Root directory containing folders to process')
    args = parser.parse_args()

    process_clips(args.root_path)
