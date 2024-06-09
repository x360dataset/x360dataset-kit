
import os, torch, numpy as np
from glob import glob

from torchvision.transforms.functional import rotate as torchrotate
import torchvision.transforms.functional as F


def get_framenpy(self, mp4file, size=256, mmap_mode='r'):
    npy = self.get_framefolder(mp4file) + f'-{size}.npy'
    if not os.path.exists(npy):
        print("=== npy not exists:", npy)

    try:
        mem = np.load(npy, mmap_mode=mmap_mode)
    except:
        print("=== npy load fails:", npy)
        mem = None

    return mem



def read_frames_fromnpy(self, npy, frame_stride=1, frame_number=5,
                frame_offset=0, clip_len=1, total_clip_len=1,
                clip_interval=1, final_img_size=224, mode="train", resizer=None, transform=None):

        # , size=final_img_size
        total_frame = npy.shape[0]
        frame_id_list = list(np.arange(total_frame))

        one_clip_len = frame_stride * frame_number + clip_interval

        # if the first clip
        if frame_offset < clip_interval:
            frame_offset = np.random.randint(0,
                           np.maximum(clip_interval, total_frame - one_clip_len * total_clip_len))

        # Not enough frames for one clip
        if total_frame < frame_offset + one_clip_len:
            frame_offset = total_frame - one_clip_len

        frame_offset = 0 if mode != "train" else frame_offset

        final_frame = frame_stride * frame_number + frame_offset

        num_of_frame_shouldbe = (final_frame-frame_offset)//frame_stride
        
        if frame_offset > len(frame_id_list):
            num_of_frame_shouldbe = (final_frame-frame_offset)//frame_stride
            frame_stride =  np.maximum(len(frame_id_list) - num_of_frame_shouldbe, 0)
            
        selected_frame_list = frame_id_list[frame_offset:final_frame:frame_stride]
        
        # Pad selected frame
        if len(selected_frame_list) < num_of_frame_shouldbe:
            for i in range(num_of_frame_shouldbe - len(selected_frame_list)):
                selected_frame_list.append(selected_frame_list[-1])
                
        
        selected_frame_list.sort()

        try:
            cropinfo = self.get_crop_info(npy[0], byshortmax=final_img_size, resizer=resizer)
        except:
            print("=== get cropinfo fails:", npy.shape, selected_frame_list)

        frames = npy[selected_frame_list]
        images = self.crop_by_ratio(frames, info=cropinfo, center=(mode == 'test')).copy()

        if np.max(images) > 1:
            images = images / 255.0

        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        if mode == "train":
           images = albumentation_group(images)

        images = images.permute(1, 0, 2, 3)

        return images, frame_offset, total_frame

    
    
def albumentation_group(images):
    # horizontal flip 
    if np.random.random() < 0.5:
        images = torch.flip(images, dims=[3])
    # vertical flip 
    if np.random.random() < 0.5:
        images = torch.flip(images, dims=[2])

    if np.random.random() < 0.5:
        random_angle = np.random.randint(-15, 15)
        images = torchrotate(images, random_angle)

    if np.random.random() < 0.5:
        brightness_factor = np.random.uniform(0.8, 1.2)
        images = F.adjust_brightness(images, brightness_factor=brightness_factor)
        
    if np.random.random() < 0.5:
        contrast_factor = np.random.uniform(0.8, 1.2)
        images = F.adjust_contrast(images, contrast_factor=contrast_factor)
    
    return images