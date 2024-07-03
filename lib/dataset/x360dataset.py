import numpy as np, cv2
import sys, torch
from torch.utils.data import Dataset
import random, time
from .trans_util import get_img_trans, get_au_trans, get_ratio_resizer

from configs import get_env
cfg = get_env()   # cfg from file
reporoot = cfg.REPO_ROOT
sys.path.append(reporoot)

from lib.database import database, get_framenpy, read_frames_fromnpy, read_audio, read_at, get_audio_feat, get_video_feat
from .datasets import register_dataset
import torch.nn.functional as F

# cfg, is_training, split
@register_dataset("x360_temporal")
class x360Dataset(Dataset):
    def __init__(
            self,
            args,
            mode,     # if in training mode
            feat_stride,  # temporal stride of the feats
            num_frames,  # number of frames for each feat
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            trunc_thresh,  # threshold for truncate an action segment
            crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
            input_dim,  # input feat dim
            num_classes,  # number of action categories
            file_prefix,  # feature file prefix if any
            file_ext,  # feature file extension if any
            force_upsampling,  # force to upsample to max_seq_len
            panoramic_feat_file,
            front_view_feat_file,
            binocular_feat_file,
            filter=False
    ):
    
        assert crop_ratio == None or len(crop_ratio) == 2

        
        self.db = database(check_data=False)
        DEDUG = False
        self.args = args
        self.add_process_M()

        train_ids, val_ids, test_ids = self.db.get_train_val_test_id()
        if DEDUG:
            train_ids = train_ids[:100]
        print("== train_ids:", len(train_ids), "val_ids:", len(val_ids), "test_ids:", len(test_ids))
        if mode == "train":
            self.ids = train_ids # if is_training else test_ids
        elif mode == "test":
            self.ids = test_ids
        elif mode == "val":
            self.ids = val_ids

        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        # self.json_file = args.json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.mode = mode

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

        self.filter = filter
        self.feat_file = {"panoramic": panoramic_feat_file, "front_view": front_view_feat_file,
                          "binocular": binocular_feat_file}

        self.db_attributes = {
            'dataset_name': 'ActivityNet 1.3',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }

    def get_attributes(self):
        return self.db_attributes

    def add_process_M(self):
        self.Ms = self.args['Ms']
        self.process_key = []
        if self.Ms["panoramic"]:
            self.process_key.append("panoramic")
        if self.Ms['front_view']:
            self.process_key.append("front_view")
        if self.Ms['binocular']:
            self.process_key.append("binocular")

    def fact_vec_new(self, feats, video_item, video_length=0):
        # (325, 1024) or (86, 20102448)-> 1024, 192
        # we support both fixed length features / variable length features

        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                print("down sample: true")
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate

        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            # center the features
            # print("force sample stride: ", feat_stride)
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
            print("feat_stride in case 3:", feat_stride)

        try:
            feat_offset = 0.5 * num_frames / feat_stride
        except:
            print("num_frames:", num_frames, feat_stride, video_item['duration'], video_item['fps'])

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # print("feats in vec:", feats.shape)
        # [1024, 64]
        # video_length
        if video_length == 0:
            video_length = feats.shape[-1]

        # resize the features if needed
        # Resize the first
        if (feats.shape[-1] != video_length):  # and self.force_upsampling:
            s = feats.shape[0]
            feats = feats.unsqueeze(0).unsqueeze(0)
            # print("feats no match:", feats.shape)
            resize_feats = F.interpolate(
                feats,
                size=[s, video_length],
                mode='bilinear',
                align_corners=False
            )
            feats = resize_feats.squeeze()
            # print("after feat:", feats.shape)

        return feats, feat_stride, num_frames, feat_offset

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
        try:
            feat_offset = 0.5 * num_frames / feat_stride
        except:
            print("num_frames:", num_frames, feat_stride, video_item['duration'], video_item['fps'])

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

    def concat(self, f, a):
        if isinstance(f, type(None)):
            return a
        else:
            return torch.concat((f, a), axis=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        id = self.ids[idx]
        data = self.db.get_cutdata(id)
        fused_feats = None
        feats = None
        # v_length =
        video_length = 100
        at_length = 256
        # output = {}
        data_dict = {}

        if self.Ms['at']:
            mp4name = data['front_view'].split("/")[-1]
            at_feats = np.load(data['front_view'].replace(mp4name, "at.npy"), allow_pickle=True).astype(np.float32)
            at_feats = torch.from_numpy(at_feats)
            at_feats = at_feats.unsqueeze(-1).expand(-1, video_length)

            at_feats = at_feats.unsqueeze(0).unsqueeze(0)
            # print("feats no match:", feats.shape)
            resize_feats = F.interpolate(
                at_feats,
                size=[at_length, video_length],
                mode='bilinear',
                align_corners=False
            )
            at_feats = resize_feats.squeeze()

        for key in self.process_key:
            video_item = data[key]
            # fps, num_frames, duration = self.get_videoinfo(video_item)

            mp4name = data[key].split("/")[-1]
            # dr.get_video_feat(
            try:
                video_feats = np.load(data[key].replace(mp4name, self.feat_file[key])).astype(np.float32)
            except:
                print("video_feat read failed:", data[key].replace(mp4name, self.feat_file[key]))

            audio_feats = np.load(data[key].replace(mp4name, "audio_feat.npy")).astype(np.float32)

            # try:
            video_feats, feat_stride, num_frames, feat_offset = self.fact_vec_new(video_feats, video_item, video_length)
            # video_length = video_feats.shape[-1]
            audio_feats, _, _, _ = self.fact_vec_new(audio_feats, video_item, video_length)  # [128, 192]

            feats = video_feats  # self.concat(feats, video_feats)

            if self.Ms['audio']:
                feats = self.concat(feats, audio_feats)
            

            data_dict[f'{key}_feats'] = feats
            fused_feats = self.concat(fused_feats, feats)
            feats = None
            
        if self.Ms['at']:
            fused_feats = self.concat(fused_feats, at_feats)
            
        temporal_label = data['temporal_label']
        segments = np.asarray(data['segments'])
        constant = 10.001

        filter_small_segments = True
        filter_time_span = 2  # 2second
        if filter_small_segments and len(temporal_label) > 0:
            s_diff = segments[:, 1] - segments[:, 0]
            temporal_label = np.array(temporal_label)[s_diff > filter_time_span]
            segments = segments[s_diff > filter_time_span]

        # Handle Segmentations
        if len(temporal_label) > 0:  # Vacant
            seg = np.asarray([[i[0] % constant, i[1] % constant] for i in segments], dtype=np.float16)

            segments = torch.from_numpy(
                seg * 25  # data['fps']
            )

            try:
                labels = torch.from_numpy(np.array(temporal_label))
            except:
                labels = torch.from_numpy(np.array(self.db.temporal_mapping[temporal_label]))

            for i in labels:
                i.to(torch.int64)
        else:
            segments = torch.from_numpy(np.array([[0.0, 0.0]], dtype=np.float32))
            labels = torch.from_numpy(np.array([0]))

        if idx % (self.__len__() - 1) == 0:
            print("feats shape:", fused_feats.shape)  # 3072, 100;  Full 11100

        # return a data dict
        temp_dict = {'video_id': id,  # video_item['id'],
                     'feats': fused_feats,  # C x T
                     'segments': segments,  # N x 2
                     'labels': labels,  # N
                     'fps': 25,  # data['fps'],
                     'duration': data['duration'],
                     'feat_stride': self.args['dataset']['feat_stride'],
                     'feat_num_frames': num_frames}

        return data_dict | temp_dict




@register_dataset("x360cls")
class x360Dataset_cls(Dataset):
    def __init__(self, args, mode='train', max_cut_number=6, use_video_frames=5,
                 frame_stride=1, max_sr=1, clip_len=1, clip_interval=0):

        self.args = args
        self.mode = mode
        self.data_list = []
        self.max_cut_number = max_cut_number    # how many video cut in one data sample
        self.use_video_frames = use_video_frames
        self.frame_stride = frame_stride  # control the sample ratio, or speed of video

        self.max_sr = max_sr       # max speed ratio
        

        self.clip_len = clip_len         # length of video clip
        self.clip_interval = clip_interval

        self.final_img_size = 224
        self.resize_img_size = 256

        self.Ms = self.args.Ms
        try:
            self.read_feature = self.args.read_feature
        except:
            self.read_feature = False

        if mode == "total":
            self.ids = self.db.get_cutlist()
        else:
            self.ids = self.train_ids if mode == 'train' else self.val_ids if mode == 'val' else self.test_ids

        # self.ids = db.get_cutlist()
        self.img_resizer = get_ratio_resizer(self.resize_img_size)
        self.img_trans = get_img_trans(mode)
        self.au_trans, self.spec_trans = get_au_trans(mode)

        self.add_process_M()

    def add_process_M(self):
        self.process_key = []
        if self.Ms["panoramic"]:
            self.process_key.append("panoramic")
        if self.Ms['front_view']:
            self.process_key.append("front_view")
        if self.Ms['binocular']:
            self.process_key.append("binocular")

    def __len__(self):
        return len(self.ids)

    def choose_cut_data(self, data, selected_cut):
        # Get cut number of the many cuts of the same data sample
        data['panoramic'] = data['panoramic'][selected_cut]
        data['front_view'] = data['front_view'][selected_cut]

        sc = np.minimum(selected_cut, len(data['binocular']) - 1)
        data['binocular'] = data['binocular'][sc]
        sc = np.minimum(selected_cut, len(data['monocular']) - 1)
        data['monocular'] = data['monocular'][sc]

        return data

    def get_videoinfo(self, video_path):
        vid = cv2.VideoCapture(video_path)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        frame_number = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        try:
            duration = frame_number / fps
        except:
            print("fps=0")
            duration = 1

        return fps, frame_number, duration

    def get_features_item_mean(self, data, Output_dict):

        #
        # video_length = 20
        ratio = np.random.random() * 0.8

        # Binocular/Front_view/Panoramic
        # 10s
        for key in self.process_key:  #

            video_feat = get_video_feat(data[key])  # [x, 1024]
            print("video_feat.shape:", video_feat.shape)

            frame_start = np.floor(ratio * video_feat.shape[0])
            frame_end = np.ceil((ratio + 0.2) * video_feat.shape[0])
            if frame_end == frame_start:
                frame_end += 1
            video_feat = video_feat[frame_start: frame_start + frame_end]
            if len(video_feat.shape) > 1:
                video_feat = np.mean(video_feat)

            audio_feat = get_audio_feat(data[key])  # [y, 128]
            print("audio_feat.shape:", audio_feat.shape)
            audio_start = np.floor(ratio * audio_feat.shape[0])
            audio_end = np.ceil((ratio + 0.2) * audio_feat.shape[0])
            if audio_end == audio_start:
                audio_end += 1

            audio_feat = audio_feat[audio_start: audio_end]
            if len(audio_feat.shape) > 1:
                audio_feat = np.mean(audio_feat)

            Output_dict[key] = {"video": video_feat, "audio": audio_feat}

        if self.Ms['at']:
            at = read_at(data['front_view'], max_len=250)  # [z, 128]
            at_start = np.floor(ratio * at.shape[0])
            at_end = np.ceil((ratio + 0.2) * at.shape[0])
            if at_end == at_start:
                at_end += 1
            at = at[at_start: at_end]
            if len(at.shape) > 1:
                at = np.mean(at)

            Output_dict['at'] = np.expand_dims(at, 0)

        return Output_dict

    def get_features_item_all_length(self, data, Output_dict, mean=False):

        # Binocular/Front_view/Panoramic
        # 10s
        for key in self.process_key:  #

            video_feat = get_video_feat(data[key])  # [x, 1024]
            # print("video_feat.shape:", video_feat.shape)
            # (x, 1024)
            if len(video_feat.shape) > 1 and mean:
                video_feat = np.mean(video_feat, axis=0)

            audio_feat = get_audio_feat(data[key])  # [y, 128]
            # (y, 128)

            if len(audio_feat.shape) > 1 and mean:
                audio_feat = np.mean(audio_feat, axis=0)

            Output_dict[key] = {"video": video_feat, "audio": audio_feat}

        if self.Ms['at']:
            at = read_at(data['front_view'], max_len=250)  # [z, 128]

            if len(at.shape) > 1 and mean:
                at = np.mean(at)

            Output_dict['at'] = at #np.expand_dims(at, 0)

        return Output_dict

    def get_features_item(self, data, Output_dict):
        pass

    def __getitem__(self, idx):

        id = self.ids[idx]
        data = self.db.get_cutdata(id)
        cls_name = data[self.db.category_key]  # or old_category_key
        cls = self.db.cls_mapping[cls_name]
        Output_dict = {"cls": cls, "cls_name": cls_name}

        if self.read_feature:
            Output_dict = self.get_features_item_all_length(data, Output_dict, mean=True)
            return Output_dict

            # return data
            # |   ├── video_feat.npy         (x, 1024) x = 50~ 70
            #    ├── audio_feat.npy         (x, 128)  x = 10~ 20

        frame_sr = self.frame_stride if not self.ssl_speed else random.randint(1, self.max_sr)
        if self.ssl_speed:
            Output_dict['ssl_sr'] = frame_sr - 1

        if self.clip_len > 1:
            clip_ids = list(range(self.clip_len))  # clips id
            random.shuffle(clip_ids)

            Output_dict['ssl_clip'] = clip_ids

        for key in self.process_key:
            frame_offset = 0

            if self.clip_len > 1:
                Output_dict[key] = {"f": [], "a": []}
                clip_frames = []
                clip_left_audios, clip_right_audios = [], []

            npy = get_framenpy(data[key])

            DEBUG = False
            for clip in range(self.clip_len):

                # if DEBUG:
                start_time = time.time()

                frames, frame_offset, frame_num = read_frames_fromnpy(npy, frame_stride=frame_sr,
                                                                         frame_number=self.use_video_frames,
                                                                         frame_offset=frame_offset,
                                                                         total_clip_len=self.clip_len,
                                                                         clip_len=self.clip_len - clip,
                                                                         clip_interval=self.clip_interval,
                                                                         mode=self.mode,
                                                                         final_img_size=self.final_img_size,
                                                                         resizer=self.img_resizer,
                                                                         transform=self.img_trans)

                if time.time() - start_time > 5 or DEBUG:  # 5s time out
                    print("TIMEOUT get frames time: ", time.time() - start_time)
                start_time = time.time()

                # 130, 255]
                audios = read_audio(data[key], rate=16000,
                                       frame_sr=frame_sr,
                                       frame_offset=frame_offset,
                                       fps=25,
                                       frame_num=self.frame_stride * self.use_video_frames,
                                       max_len=(129, 80), mode=self.mode,
                                       au_transform=self.au_trans,  # TODO Test au_transform
                                       spec_transform=self.spec_trans)

                if time.time() - start_time > 5 or DEBUG:  # 5s time out
                    print("TIMEOUT get audios time: ", time.time() - start_time)

                # audios is a list: [left_audio, right_audio]
                frame_offset += self.clip_interval

                # frames is a list: [f1, f2, ..., fn]
                if self.clip_len > 1:
                    clip_frames.append(frames)
                    clip_left_audios.append(audios[0])
                    clip_right_audios.append(audios[1])
                else:
                    Output_dict[key] = {"f": frames, "a": audios}

            if self.clip_len > 1:
                frames = torch.stack([clip_frames[i] for i in clip_ids])
                left_audio = torch.Tensor(np.stack([clip_left_audios[i] for i in clip_ids]))
                right_audio = torch.Tensor(np.stack([clip_right_audios[i] for i in clip_ids]))
                audios = [left_audio, right_audio]

                Output_dict[key] = {"f": frames, "a": audios}

        if self.Ms['at']:
            at = read_at(data['front_view'], max_len=250)
            Output_dict['at'] = np.expand_dims(at, 0)

        if self.clip_len > 1:
            Output_dict['ssl_clip'] = torch.Tensor(np.array(Output_dict['ssl_clip']))

        return Output_dict









