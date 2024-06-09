import albumentations as A
import audiomentations as Au
from albumentations.pytorch import ToTensorV2

def get_img_trans(mode):
    if mode == 'train':
        transform = A.Compose([
            # A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                       contrast_limit=(-0.1, 0.1), p=0.5),
            A.GaussNoise(),
            ToTensorV2(),
        ])

    else:
        transform = A.Compose([
            ToTensorV2(),
        ])

    return transform

def get_au_trans(mode):
    try:
        au_shift = Au.Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
    except:
        au_shift = Au.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)

    if mode == 'train':
        au_transform = Au.Compose([
            Au.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            Au.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            Au.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            au_shift,

        ])

        spec_transform = Au.SpecCompose([
            # Au.SpecChannelShuffle(p=0.5),
            Au.SpecFrequencyMask(p=0.5),
        ])

    else:
        au_transform = Au.Compose([])
        spec_transform = Au.SpecCompose([])

    return au_transform, spec_transform

def get_stand_resizer(size):
    return A.Compose([
        A.SmallestMaxSize(size, interpolation=3)
    ])

def get_ratio_resizer(size, ratio=2):
    return A.Compose([
        A.Resize(size*ratio, int(size*ratio), interpolation=3),
    ])