import os
import shutil
from tqdm import tqdm

base_path = '/bask/projects/j/jiaoj-rep-learn/360XProject/ICCV_Workshop_Data/Han/unanonymized_published_data'
output_base = '/bask/projects/j/jiaoj-rep-learn/360XProject/ICCV_Workshop_Data/Han/Ready_data'

# Create target folders if they don't exist
for subfolder in ['binocular', 'monocular', 'panoramic', 'third_person']:
    os.makedirs(os.path.join(output_base, subfolder), exist_ok=True)

# Iterate over all folders inside base_path
for folder in tqdm(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)

    if not os.path.isdir(folder_path):
        continue  # Skip files

    ### --- Handle 360 folder ---
    panoramic_path = os.path.join(folder_path, "360", "360_panoramic.mp4")
    third_person_path = os.path.join(folder_path, "360", "front_view.mp4")

    if os.path.exists(panoramic_path):
        target_path = os.path.join(output_base, "panoramic", f"{folder}.mp4")
        shutil.copy2(panoramic_path, target_path)
        print(f"Copied panoramic for {folder}")

    if os.path.exists(third_person_path):
        target_path = os.path.join(output_base, "third_person", f"{folder}.mp4")
        shutil.copy2(third_person_path, target_path)
        print(f"Copied third_person for {folder}")

    ### --- Handle Snapchat folders ---
    snapchat_path = os.path.join(folder_path, "Snapchat")

    if os.path.exists(snapchat_path) and os.path.isdir(snapchat_path):
        for clip in os.listdir(snapchat_path):
            clip_path = os.path.join(snapchat_path, clip)
            if not os.path.isdir(clip_path):
                continue

            mono_path = os.path.join(clip_path, "mono.mp4")
            stereo_path = os.path.join(clip_path, "stereo.mp4")

            if os.path.exists(mono_path):
                target_dir = os.path.join(output_base, "monocular", folder)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, f"{clip}.mp4")
                shutil.copy2(mono_path, target_path)
                print(f"Copied monocular {clip} for {folder}")

            if os.path.exists(stereo_path):
                target_dir = os.path.join(output_base, "binocular", folder)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, f"{clip}.mp4")
                shutil.copy2(stereo_path, target_path)
                print(f"Copied binocular {clip} for {folder}")
