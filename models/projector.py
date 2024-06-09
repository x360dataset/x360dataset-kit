from .models import r21d, i3d
from .models import vggish, resnet1d
#  r3d, c3d, s3d_g, inceptioni3d,

from torch import nn, optim
import torch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from .optimize import GradualWarmupSchedulerV3


def expand(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class get_head(nn.Module):
    def __init__(self, feature_size=512, num_classes=1, dropout_prob=0.15):
        super(get_head, self).__init__()

        # dropout_prob = 0.15  # 0.15
        fusion_size = 256

        self.fc_fusion = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(feature_size, fusion_size)
        )

        self.fc_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_size, num_classes)
        )

        self.feature_size = feature_size

        self.attention = nn.Sequential(
            # torch.nn.Dropout(dropout_prob),
            nn.Linear(fusion_size, fusion_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc_fusion(x)
        feat = self.attention(x) * x  # expand(
        return self.fc_head(feat)  # , self.aux_head(feat)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def squeeze(x):
    return x.squeeze(3).squeeze(3).mean(2)


def concat(f, x):
    # x shape: torch.Size([30, 512, 8])
    # f shape: torch.Size([30, 1024, 1, 1, 1])

    if f == None:
        return x
    elif x == None:
        return f
    else:
        try:
            if len(f.shape) > 3:
                # return torch.cat((f, expand(x)), dim=1)
                return torch.cat((squeeze(f), x), dim=-1)
            return torch.cat((f, x), dim=-1)
        except:
            print("x shape:", x.shape)
            print("f shape:", f.shape)


class Projector(nn.Module):
    def __init__(self, args, num_classes=1):
        super(Projector, self).__init__()

        self.args = args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # args.num_classes = 21 if args.task.upper() == 'CLS' else 1
        final_embed = 128
       
        videofeature_size = 1024
        feature_size = videofeature_size
        number_of_videomodality = 0
        number_of_videomodality += 1 if args.Ms['front_view'] else 0
        number_of_videomodality += 1 if args.Ms['binocular'] else 0
        number_of_videomodality += 1 if args.Ms['panoramic'] else 0
        feature_size = feature_size * number_of_videomodality

        print("=== number_of_videomodality: ", number_of_videomodality)
        dropout_prob = 0.25
        
        self.video_model = nn.Sequential(
            # nn.Dropout(args.droprate),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512,  256),
            nn.PReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(256, final_embed).to(device)).to(device)
       
        
    
        
        parameters = self.video_model.parameters()

        if args.Ms['audio']:
            feature_size += 128 * number_of_videomodality
            self.audio_model = nn.Sequential(
                # nn.Dropout(args.droprate),
                nn.LayerNorm(128),
                nn.Linear(128, 512),
                nn.PReLU(),
                nn.Linear(512, 256),
                nn.PReLU(),
                nn.Dropout(dropout_prob),

                nn.Linear(256, final_embed).to(device)

            ).to(device)
            
            parameters = list(parameters) + list(self.audio_model.parameters())
            self.audio_model.apply(weight_init)
            self.audio_model.to(device)
            self.audio_model.train()


        if args.Ms['at']:
            at_feature_size = 255
            feature_size += at_feature_size
            at = nn.Sequential(
                # nn.Dropout(args.droprate),
                nn.LayerNorm(255),
                nn.Linear(255, 512),
                nn.PReLU(),
                nn.Linear(512, 256),
                nn.PReLU(),
                nn.Dropout(dropout_prob),

                nn.Linear(256, final_embed).to(device)

            ).to(device)

            parameters = list(parameters) + list(at.parameters())
            at.apply(weight_init)
            at.to(device)
            at.train()
            self.at_model = at


        self.optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)  # 0.005
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        warmup_epo = 1
        warmup_factor = 10
        epochs = 30
        T_max = epochs - warmup_epo - 1
        min_lr = 1e-6

        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=T_max,
                                           eta_min=min_lr, last_epoch=-1)

        self.scheduler_warmup = GradualWarmupSchedulerV3(self.optimizer, multiplier=10,
                                                         total_epoch=warmup_epo,
                                                         after_scheduler=self.scheduler)



        print("=====  FEATURE SIZE", feature_size, '  =====')

    def get_feature_modality_wise(self, args, data_input, tags, device):
        # if args.clip_number > 1:

        modality = 1 + 1 * args.Ms['audio'] + 1 * args.Ms['at']

        # print("modality in get_feature_modality_wise:", modality)

        self.optimizer.zero_grad()
        feat = None
        video_feat = None
        audio_feat = None
        # print("tags:", tags)
        for tag in tags:
            # print("video input shape:", data_input[tag]['video'].shape)
            feat = self.video_model(data_input[tag]['video'])  # outputs  n, 1024, 1, 1, 1
            video_feat = concat(feat, video_feat)

            if args.Ms['audio']:
                # print("audio input shape:", data_input[tag]['audio'].shape)

                feat = self.audio_model(data_input[tag]['audio'])  # outputs  n, 128, 1, 1, 1
                audio_feat = concat(feat, audio_feat)

        #
        if args.Ms['at']:
            at_feat = self.at_model(data_input['at'])
            at_feat = at_feat.repeat(1, len(tags))

        if modality == 1:
            return video_feat
        elif modality == 2:
            return video_feat, audio_feat
        elif modality == 3:
            return video_feat, audio_feat, at_feat

    def forward_head(self, feat):

        if len(feat.shape) > 3:
            feat = feat.view(-1, feat.shape[-1])

        B, C, T = feat.shape
        feat = feat.view(B * C, T)
        out = self.head(feat)
        return out  # out.view(B, C, -1)



    def eval_model(self):
        self.video_model.eval()

        if self.args.Ms['audio']:
            self.audio_model.eval()


        if self.args.Ms['at']:
            self.at_model.eval()



    def train_model(self):
        self.video_model.train()

        if self.args.Ms['audio']:
            self.audio_model.train()

        if self.args.Ms['at']:
            self.at_model.train()