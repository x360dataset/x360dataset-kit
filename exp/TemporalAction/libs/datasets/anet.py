import os
import json
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations
import pandas as pd

@register_dataset("anet")
class ActivityNetDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        video_feat_folder,      # folder for features
        audio_feat_folder,      # folder for features
        json_file,        # json file for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        filter = False
    ):
        # file path
        assert os.path.exists(video_feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        
        # anet
        self.csv = pd.read_csv("data/anet.csv")['video_id']
        self.video_feat_folder = video_feat_folder
        self.audio_feat_folder = audio_feat_folder
        self.use_hdf5 = '.hdf5' in video_feat_folder
        
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        
        
        # proposal vs action categories
        assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db
        self.label_dict = label_dict

        self.filter = filter
        
        if filter:
            self.process_data_list()
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'ActivityNet 1.3',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }

    def process_data_list(self):
        # print("self.data_list:", self.data_list)  # v_'id'
        # print("self.csv:", self.csv.to_list()) 
        dict_db = []
        print("Filter = True")
        
        for i, d in enumerate(self.data_list):
            n = "v_" + d['id']
            
            if n in self.csv.to_list():
                dict_db.append( d )
                
        self.data_list = tuple(dict_db)  

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)  # /bask/projects/j/jiaoj-3d-vision/Hao/360x/360data/ActivityNet/annotations/anet1.3_i3d_filtered.json
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:  # act
                    # print("act:", act)
                    label_dict[act['label']] = act['label_id']   # 

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            
            # skip the video if not in the split
            # if value['subset'].lower() not in self.split:
            #     continue
            
            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    if self.num_classes == 1:
                        labels[idx] = 0
                    else:
                        labels[idx] = label_dict[act['label']]
            else:
                segments = None
                labels = None
            
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        
        # print("dict_db, label_dict:", dict_db, label_dict)
        
        return dict_db, label_dict


    def fact_vec(self, feats, video_item):
        # (325, 2048) or (86, 2048)-> 2048, 192
        # we support both fixed length features / variable length features

        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate

        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            # center the features
            num_frames = feat_stride

        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # resize the features if needed
        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)

        return feats, feat_stride, num_frames, feat_offset

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        
        # load features
        if self.use_hdf5:
            with h5py.File(self.video_feat_folder, 'r') as h5_fid:
                feats = np.asarray(
                    h5_fid[self.file_prefix + video_item['id']][()],
                    dtype=np.float32
                )
        else:
            filename = os.path.join(self.video_feat_folder,
                                    self.file_prefix + video_item['id'] + self.file_ext)

            feats = np.load(filename).astype(np.float32)
            audio_file = os.path.join(self.audio_feat_folder,
                                    self.file_prefix + video_item['id'] + self.file_ext)
            audio_feats= np.load(audio_file).astype(np.float32)

        # (4741, 128) (296, 2048)
        # (214, 128) (11, 2048)

        # print("load in video feats:", audio_feats.shape, feats.shape)

        feats, feat_stride, num_frames, feat_offset = self.fact_vec(feats, video_item)
        audio_feats, _, _, _  = self.fact_vec(audio_feats, video_item)   # [128, 192]

        # print("load in video feats:", audio_feats.shape, feats.shape)
        # torch.Size([128, 192]) torch.Size([2048, 192])

        # make_meta_arch: name = LocPointTransformer
        # make_generator: name = point

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            
            
            labels = torch.from_numpy(video_item['labels'])
            
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training and self.filter:
                vid_len = feats.shape[1] + feat_offset  # 192.5
                valid_seg_list, valid_label_list = [], []
                
                for seg, label in zip(segments, labels):
                    
                    if seg[0] >= vid_len and self.filter:
                        # skip an action outside of the feature map
                        # print("continue:", seg[0],  vid_len )
                        continue
                    
                    # skip an action that is mostly outside of the feature map
                    if filter:
                    
                        ratio = (
                            (min(seg[1].item(), vid_len) - seg[0].item())
                            / (seg[1].item() - seg[0].item())
                        )
                    else:
                        ratio = (
                            (min(seg[1].item(), vid_len) - seg[0].item())
                            / (seg[1].item() - seg[0].item())
                        )
                        
                        print("ratio:", ratio, " self.trunc_thresh:", self.trunc_thresh)
                        
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                        
                # print("valid_seg_list:", valid_seg_list)
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        # print("video feats:", feats.shape, audio_feats.shape)   # 2048, 192
        
        # print("segments:", segments)
        
        
        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'audio_feats'     : audio_feats,
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        # no truncation is needed
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict
