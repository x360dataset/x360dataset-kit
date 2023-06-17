"""VCOPN"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch.nn.functional as F


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
                return torch.cat((squeeze(f), x), dim=1)
            return torch.cat((f, x), dim=1)
        except:
            print("x shape:", x.shape)
            print("f shape:", f.shape)



class VCOPN(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, args, base_network, feature_size, tuple_len,num_classes):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN, self).__init__()

        self.args = args
        self.base_network = base_network
        if args.use_clip:
            self.tag = 'clip'
        elif args.use_front:
            self.tag = 'front'
        elif args.use_360:
            self.tag = '360'

        feature_s = 0


        if args.use_audio:
            self.audio_net = base_network['audio_model']
            feature_s += 128
        if args.use_directional_audio:
            self.at_net = base_network['at_model']
            feature_s += 128

        self.visual_model = base_network['visual_model']
        
        
        dropout_prob = 0.15  # 0.15

        
        # self.fc_head = nn.Sequential(
        #             # nn.Dropout(dropout_rate),
        #             nn.Linear(feature_size, num_classes)
        #         )

        self.feature_size = feature_size
        pair_num = int(tuple_len*(tuple_len-1)/2)



        self.attention = nn.Sequential(
            torch.nn.Dropout(dropout_prob),
            nn.Linear(512*pair_num+ feature_s, 512*pair_num + feature_s),
            nn.Sigmoid()
        )


        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)  # 3!

        self.fc7 = nn.Linear(self.feature_size*2, 512)


        self.fc8 = nn.Linear(512*pair_num + feature_s, self.class_num)

        # self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
    def forward(self, sample):
            # tuple_clips = data[tag]["f"]
        
        fin = sample[self.tag]["f"].to(self.device)

        # VCOP
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = fin[:, i, :, :, :, :]
            v = self.visual_model(clip)

            if self.args.model != 'i3d':
                (_, C, T, H, W) = v.size()
                B = v.size()[0] #// 3
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)
                v = F.adaptive_avg_pool3d(v, 1)
                v = torch.flatten(v, 1)
        
            
            f.append(v)
            # print("f size:", f1.shape)
            
        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))
        
        
        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]

        feat = torch.cat(pf, dim=1)



        if self.args.use_audio:
            left_audio = sample[self.tag]["a"].to(self.device, dtype=torch.float)
            # right_audio = sample[tag]["a"][1].to(device, dtype=torch.float)
            a_feat = self.audio_net(left_audio)
            feat = concat(feat, a_feat)

        if self.args.use_directional_audio:
            at = sample['at'].to(self.device, dtype=torch.float)
            at_feat = self.at_net(at)
            feat = concat(feat, at_feat)


        feat = self.attention(feat) * feat

        out_dict = self.fc8(feat)
        h = out_dict # [0])  # logits

        return h


class VCOPN_RNN(nn.Module):
    """Video clip order prediction with RNN."""
    def __init__(self, base_network, feature_size, tuple_len, hidden_size, rnn_type='LSTM'):
        """
        Args:
            feature_size (int): 1024
        """
        super(VCOPN_RNN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.feature_size, self.hidden_size)
        elif self.rnn_type == 'GRU':
            self.gru = nn.GRU(self.feature_size, self.hidden_size)
        
        self.fc = nn.Linear(self.hidden_size, self.class_num)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))

        inputs = torch.stack(f)
        if self.rnn_type == 'LSTM':
            outputs, (hn, cn) = self.lstm(inputs)
        elif self.rnn_type == 'GRU':
            outputs, hn = self.gru(inputs)

        h = self.fc(hn.squeeze(dim=0))  # logits

        return h
