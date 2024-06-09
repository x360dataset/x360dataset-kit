import cv2, numpy as np
import albumentations as A


def get_crop_info(self, img, byshortmax=224, label=None, resizer=None):
    H, W, C = img.shape

    Hcrop = byshortmax
    Wcrop = int(2 * byshortmax)

    rnd_h = np.random.randint(0, max(0, H - Hcrop))  # loc
    rnd_w = np.random.randint(0, max(0, W - Wcrop))

    info = {"h_size": Hcrop, "w_size": Wcrop,
            "rnd_h": rnd_h, "rnd_w": rnd_w}

    return info


def crop_by_ratio(self, img, center=False, info=None):
    if center:
        crop = A.Compose([
            A.CenterCrop(height=info['Hcrop'], width=info['Wcrop'], always_apply=True),
        ])
        cropped = crop(image=img)['image']

    else:
        # BHWC
        cropped = img[:, info['rnd_h'] : info['rnd_h'] + info['h_size'],
                         info['rnd_w'] : info['rnd_w'] + info['w_size'], : ]

    return cropped


def load_frame(self, frame_path, resizer=None):
    img = cv2.imread(frame_path)  
    
    if type(img) is np.ndarray:
        if resizer:
            img = resizer(image=img)["image"]
    else:
        print("WARNING: img frame is not found: ", frame_path)
        print(img)
        
    
    return img