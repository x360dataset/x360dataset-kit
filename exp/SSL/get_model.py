
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
from models.vcopn import VCOPN
import torch

from models.res3d import r3d_18 
from models.i3d import I3D
from models import vggish

from models import i3d
from models import vggish, resnet1d

from torch import nn, optim
import torch


class get_criterian(nn.Module):
    def __init__(self):
        super(get_criterian, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()

    def forward(self, pred, target, aux_pred):
        CE = self.CE(pred, target)

        MSE = self.MSE(aux_pred.to(torch.float32),
                       target.unsqueeze(-1).to(torch.float32))

        return CE #+ (MSE/ (MSE/CE + 1e-5).detach())


def expand(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def get_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dict = {}

    num_classes = args.tl

    if args.model == 'r21d':
        model = r21d.R2Plus1DNet(num_classes=num_classes)

    elif args.model == 'r3d':
        feature_size = 512
        model = r3d_18(pretrained=False, progress=True) # r3d.R3DNet(num_classes=num_classes)

    elif args.model == 'c3d':
        model = c3d.C3D(num_classes=num_classes)

    elif args.model == 's3d':
        model = s3d_g.S3D(num_classes=num_classes, space_to_depth=False)

    elif args.model == 'i3d':
        model = i3d.I3D(num_classes=num_classes, dropout_prob=0)
        feature_size = 1024

    if args.use_audio:
        audio_model = vggish.VGGish(pretrained=False)   # 128
        out_dict['audio_model'] = audio_model

    if args.use_directional_audio:
        at = resnet1d.resnet18_1d(modality='audio')      # 128
        out_dict['at_model'] = at


    # 4. multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    out_dict.update({'visual_model': model})

    return VCOPN(args, base_network=out_dict, feature_size=feature_size, num_classes=num_classes, tuple_len=args.tl).to(device)