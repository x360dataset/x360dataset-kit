import sys, os, cv2

import numpy as np
import albumentations as A

def get_stand_resizer(size):
    return A.Compose([
        A.SmallestMaxSize(size, interpolation=3)
    ])

def load_frame(frame_path, resizer=None):
    img = cv2.imread(frame_path)  # , cv2.IMREAD_GRAYSCALE)

    if type(img) is np.ndarray:

        if resizer:
            img = resizer(image=img)["image"]
    else:
        print("DEDEBUG: img is not ndarray: ", frame_path)
        print(img)

    return img
def get_framefolder(mp4file):
    name = mp4file.split("/")[-1]
    return mp4file.replace(name, "frames")

from glob import glob
from tqdm import tqdm


def packs_frame_folder(mp4files, smallestsize = 256):

    for mp4file in mp4files:
        folder = get_framefolder(mp4file)
        files = sorted(list(glob(folder + '/*.jpg')))

        if len(files) == 0:
            print("=== No files found in: ", folder)
            continue

        resizer = get_stand_resizer(smallestsize)

        save_npy = folder + f'-{smallestsize}.npy'   # frames.np4
        if os.path.exists(save_npy):
            # print("=== Already exists: ", save_npy)
            continue

        print("=== preparing: ", save_npy)

        for idx, f in tqdm(enumerate(files)):
            img = load_frame(f, resizer=resizer)
            if idx == 0:
                final_array = np.zeros((len(files), img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)

            final_array[idx] = img

        print("Final array shape: ", final_array.shape)
        print("=== save to: ", save_npy)
        np.save(save_npy, final_array)