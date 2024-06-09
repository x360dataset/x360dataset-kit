from .models import r21d, i3d
from .models import vggish, resnet1d
#  r3d, c3d, s3d_g, inceptioni3d,

from torch import nn, optim
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from .optimize import GradualWarmupSchedulerV3


def expand(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class TransformerEncoderOneDimension(nn.Module):
    def __init__(self, channel_dim, embedding_dim, latent_dim, num_heads, num_layers):
        super(TransformerEncoderOneDimension, self).__init__()

        # Transformer Encoder层
        encoder_layers = TransformerEncoderLayer(channel_dim, num_heads, latent_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # 全连接层，用于将输入的 channel_dim 转换为 embedding_dim
        self.fc = nn.Linear(channel_dim, embedding_dim)

    def forward(self, x):
        # x 的初始形状: (B, C, 3584)

        x = self.transformer_encoder(x)
        # 通过全连接层
        x = self.fc(x)
        # 通过 Transformer Encoder

        return x


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
            # nn.Linear(fusion_size, num_classes)
            nn.Linear(feature_size, num_classes)
        )

        self.feature_size = feature_size

        self.attention = nn.Sequential(
            # torch.nn.Dropout(dropout_prob),
            nn.Linear(fusion_size, fusion_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.fc_fusion(x)
        # feat = self.attention(x) * x  # expand(
        # return self.fc_head(feat)  # , self.aux_head(feat)
        # print(x.shape)
        return self.fc_head(x)


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
    else:
        try:
            if len(f.shape) > 3:
                # return torch.cat((f, expand(x)), dim=1)
                return torch.cat((squeeze(f), x), dim=-1)
            return torch.cat((f, x), dim=-1)
        except:
            print("x shape:", x.shape)
            print("f shape:", f.shape)


class Trainer(nn.Module):
    def __init__(self, args, num_classes=1):
        super(Trainer, self).__init__()

        self.args = args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # args.num_classes = 21 if args.task.upper() == 'CLS' else 1
        self.num_classes = num_classes



        number_of_videomodality = 0
        number_of_videomodality += 1 if args.Ms['front_view'] else 0
        number_of_videomodality += 1 if args.Ms['binocular'] else 0
        number_of_videomodality += 1 if args.Ms['panoramic'] else 0

        videofeature_size = 1024
        feature_size = videofeature_size


        self.models_key = []
        self.models = {}
        parameters = []


        if args.Ms['front_view']:
            self.models_key.append('front_view')
        elif args.Ms['binocular']:
            self.models_key.append('binocular')
        elif args.Ms['panoramic']:
            self.models_key.append('panoramic')

        for key in self.models_key:
            if args.model == 'r21d':
                self.model = r21d.R2Plus1DNet(num_classes=1)
            elif args.model == 'i3d':
                self.models[key] = i3d.I3D(dropout_prob=args.droprate)  # 1024
                self.models[key].apply(weight_init)

                parameters = list(parameters) + list(self.models[key].parameters())


                self.models[key].apply(weight_init)
                self.models[key].to(device)
                self.models[key].train()

                video_fc = nn.Sequential(
                    nn.LayerNorm(1024),
                ).to(device)
                video_head = nn.Linear(1024, args.num_classes).to(device)


                self.models[key+'_fc'] = video_fc
                self.models[key+'_head'] = video_head

                parameters = list(parameters) + list(video_fc.parameters()) + \
                             list(video_head.parameters())

        feature_size = feature_size * number_of_videomodality

        self.video_fc = nn.Sequential(
            # nn.Dropout(args.droprate),
            nn.LayerNorm(1024),
        ).to(device)
        self.video_head = nn.Linear(1024, args.num_classes).to(device)
        parameters = list(parameters) + list(self.video_fc.parameters())
        parameters = list(parameters) + list(self.video_head.parameters())

        if args.Ms['audio']:
            audio_feature = 128
            feature_size += 128 * number_of_videomodality
            audio_model = vggish.VGGish(pretrained=False, dropout_prob=args.droprate)  # 128
            parameters = list(parameters) + list(audio_model.parameters())

            audio_model.apply(weight_init)
            audio_model.to(device)
            audio_model.train()

            self.audio_model = audio_model
            self.audio_fc = nn.Sequential(
                nn.Dropout(args.droprate),
                nn.LayerNorm(128),
            ).to(device)
            self.audio_head = nn.Linear(128, args.num_classes).to(device)

            self.video_attention = nn.Sequential(
                # torch.nn.Dropout(dropout_prob),
                nn.Linear(audio_feature, videofeature_size),
                nn.Sigmoid()
            ).to(device)

            parameters = list(parameters) + list(self.video_attention.parameters())
            parameters = list(parameters) + list(self.audio_fc.parameters()) + list(self.audio_head.parameters())

        if args.Ms['at']:
            at_feature_size = 128
            feature_size += at_feature_size

            at = resnet1d.resnet18_1d(modality='audio')  # 128
            parameters = list(parameters) + list(at.parameters())
            at.apply(weight_init)
            at.to(device)
            at.train()
            self.at_model = at

            self.at_fc = nn.Sequential(
                nn.Dropout(args.droprate),
                nn.LayerNorm(128),
            ).to(device)

            self.audio_attention = nn.Sequential(
                # torch.nn.Dropout(dropout_prob),
                nn.Linear(at_feature_size, audio_feature),
                nn.Sigmoid()
            ).to(device)

            self.at_head = nn.Linear(128, args.num_classes).to(device)

            parameters = list(parameters) + list(self.audio_attention.parameters()) + \
                         list(self.at_fc.parameters()) + list(self.at_head.parameters())

        # Retrival
        if args.task == "RET":
            self.video_feat_projection = nn.Sequential(
                nn.Linear(videofeature_size, at_feature_size),
            )

        # print(f'feature_size:{feature_size}')
        embedding_dim = 256
        latent_dim = 512  # 2048..

        self.fusion = TransformerEncoderOneDimension(channel_dim=feature_size, embedding_dim=embedding_dim,
                                                     latent_dim=latent_dim, num_heads=8, num_layers=3).to(device)

        head = get_head(feature_size=embedding_dim,
                        num_classes=args.num_classes,
                        dropout_prob=args.droprate)  # Sample rate, norm to 1

        parameters = list(parameters) + list(head.parameters()) + \
                     list(self.fusion.parameters())

        head.apply(weight_init)
        head.to(device)
        head.train()

        self.head = head

        self.optimizer = optim.AdamW(parameters, lr=args.lr,
                                     weight_decay=args.wd)  # 0.005
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

        # 4. multi gpu
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        # self.model.apply(weight_init)
        # self.model.to(device)
        # self.model.train()

        print("=====  FEATURE SIZE", feature_size, '  =====')


    def eval_model(self):
        # self.model.eval()
        for key in self.models_key:
            self.models[key].eval()
            self.models[key+"_fc"].eval()
            self.models[key+"_head"].eval()

        if self.args.Ms['audio']:
            self.audio_model.eval()
            self.audio_fc.eval()
            self.audio_head.eval()
            self.video_attention.eval()

        if self.args.Ms['at']:
            self.at_model.eval()
            self.at_fc.eval()
            self.at_head.eval()
            self.audio_attention.eval()

        self.head.eval()
        self.video_fc.eval()
        self.video_head.eval()

        self.fusion.eval()


    def train_model(self):
        for key in self.models_key:
            self.models[key].train()
            self.models[key + "_fc"].train()
            self.models[key + "_head"].train()

        if self.args.Ms['audio']:
            self.audio_model.train()
            self.audio_fc.train()
            self.audio_head.train()
            self.video_attention.train()

        if self.args.Ms['at']:
            self.at_model.train()
            self.at_fc.train()
            self.at_head.train()
            self.audio_attention.train()

        self.head.train()
        self.video_fc.train()
        self.video_head.train()

        self.fusion.train()


    def get_feature(self, args, data_input, tags, device):
        # if args.clip_number > 1:

        feats_set = []

        self.optimizer.zero_grad()
        feat = None

        if args.Ms['at']:
            at_feat = self.at_model(data_input['at'])
            at_feat = self.at_fc(at_feat)

            audio_map = self.audio_attention(at_feat)

            # feats_set.append(["at", at_feat])

            at_feat = at_feat.unsqueeze(1).expand(-1, args.clip_number, -1)
            feat = concat(feat, at_feat)

        for tag in tags:

            f = self.models[tag](data_input[tag]['video'])  # outputs  n, 1024, 1, 1, 1
            f = self.models[tag+"_fc"](f)
            # print("f shape:", f.shape)    # ([6, 1024])
            feats_set.append(["f", f, "tag"])


            if args.Ms['audio']:

                audio_feat = self.audio_model(data_input[tag]['left_audio'])  # outputs  n, 128, 1, 1, 1
                #  shape '[16, 1, 128]' is invalid for input of size 768

                audio_feat = self.audio_fc(audio_feat)


                audio_feat = audio_map * audio_feat
                video_map = self.video_attention(audio_feat)
                f = video_map * f

                feats_set.append(["a", audio_feat])


                # audio_feat = audio_feat.view(data_input[tag]['left_audio'].shape[0], args.clip_number, 128)
                audio_feat = audio_feat.view(f.shape[0], args.clip_number, 128)
                # print("audio_feat shape:", audio_feat.shape)

                feat = concat(feat, audio_feat)

            f = f.view(f.shape[0] // args.clip_number, args.clip_number, 1024)  # f.shape[-1])

            feat = concat(feat, f)

        feat = self.fusion(feat)
        # print("final feat shape:", feat.shape)

        return feat, feats_set

    def get_feature_modality_wise(self, args, data_input, tags, device):
        # if args.clip_number > 1:

        modality = 1 + 1 * args.Ms['audio'] + 1 * args.Ms['at']

        print("modality in get_feature_modality_wise:", modality)

        self.optimizer.zero_grad()
        feat = None
        for tag in tags:
            video_feat = self.model(data_input[tag]['video'])  # outputs  n, 1024, 1, 1, 1
            video_feat = concat(video_feat, video_feat)

            if args.Ms['audio']:
                audio_feat = self.audio_model(data_input[tag]['left_audio'])  # outputs  n, 128, 1, 1, 1
                audio_feat = concat(audio_feat, audio_feat)

        # 
        if args.Ms['at']:
            at_feat = self.at_model(data_input['at'])
            at_feat = at_feat.unsqueeze(1).expand(-1, args.clip_number, -1)

        if modality == 1:
            return video_feat
        elif modality == 2:
            return video_feat, audio_feat
        elif modality == 3:
            return video_feat, audio_feat, at_feat

    def forward_head(self, feat):

        # different modalities

        # torch.Size([2, 3, 3456])
        if len(feat.shape) > 3:
            feat = feat.view(-1, feat.shape[-1])

        B, C, T = feat.shape
        feat = feat.view(B * C, T)
        out = self.head(feat)
        return out  # out.view(B, C, -1)

    def forward_aux_head(self, feats_set):

        # different modalities
        out_set = []
        for feat in feats_set:
            if feat[0] == "f":
                tag = feat[-1]
                out_set.append(self.models[tag](feat[1]))
            elif feat[0] == "a":
                out_set.append(self.audio_head(feat[1]))
            elif feat[0] == "at":
                out_set.append(self.at_head(feat[1]))

        return out_set  # out.view(B, C, -1)
