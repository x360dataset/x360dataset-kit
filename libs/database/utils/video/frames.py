import os, torch

import numpy as np
from glob import glob

from torchvision.transforms.functional import rotate as torchrotate
import torchvision.transforms.functional as F



def get_framefolder(self, mp4file):
    name = mp4file.split("/")[-1]
    return mp4file.replace(name, "frames")



def read_frames(self, file, frame_stride=1, frame_number=5,
                    frame_offset=0,
                    final_img_size=224, mode="train",
                    resizer=None, transform=None):

    if ".mp4" in file:
        mp4name = file.split("/")[-1]#.replace(".mp4", "")
        folder = os.path.join(file.replace(mp4name, "frames"))
    else:
        folder = file


    frame_list = glob(os.path.join(folder, '*.jpg'))

    # Less than frame_number
    if len(frame_list) < frame_offset + frame_stride * frame_number:
        for i in range(frame_offset + frame_stride * frame_number - len(frame_list)):
            frame_list.append(frame_list[-(i+2)])      # mirror-duplicates the last one

        selected_frame_list = frame_list[frame_offset::frame_stride]
    else:
        if frame_offset == 0:

            # Random frame offset (for start)
            frame_offset = np.random.randint(0, len(frame_list) - frame_stride * frame_number)

        final_frame = frame_stride * frame_number + frame_offset
        selected_frame_list = frame_list[frame_offset:final_frame:frame_stride]


    selected_frame_list.sort()


    cropinfo = self.get_crop_info(frame_list[0], byshortmax=final_img_size, resizer=resizer)
    images = torch.zeros((frame_number, 3, cropinfo['Hcrop'], cropinfo['Wcrop']))

    # images = torch.zeros((self.clip_len, 3, info['Hcrop'], info['Wcrop']))

    # every frame
    for i in range(frame_number):
        # img = Image.open(frame_list[i]).convert('RGB')
        frame = self.load_frame(selected_frame_list[i], resizer=resizer)
        # print("frame shape", frame.shape)

        frame = self.crop_by_ratio(frame, info=cropinfo,
                                    center=(mode == 'test'))


        if transform:
            frame = transform(image=frame)["image"]

        # CHW?
        images[i] = frame


    # TODO: transform the whole group?
    images = images.permute(1, 0, 2, 3)


    return images, frame_offset, len(frame_list)


