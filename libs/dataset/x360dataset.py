import numpy as np, cv2
import sys, torch
from torch.utils.data import Dataset
import random, time
from .trans_util import get_img_trans, get_au_trans, get_ratio_resizer

from configs import get_env
cfg = get_env()   # cfg from file
reporoot = cfg.REPO_ROOT
sys.path.append(reporoot)
from database import database
from database.utils import get_framenpy, read_frames_fromnpy, read_audio, read_at, get_audio_feat, get_video_feat


db = database(check_data=False)
train_ids, val_ids, test_ids = db.get_train_val_test_id()
print("== train_ids:", len(train_ids), "val_ids:", len(val_ids), "test_ids:", len(test_ids))

DEDUG = False
if DEDUG:
    train_ids= train_ids[:100]


class x360Dataset(Dataset):
    def __init__(self, args, mode='train', max_cut_number=6, use_video_frames=5,
                 frame_stride=1, max_sr=1, clip_len=1, clip_interval=0):

        self.args = args
        self.mode = mode
        self.data_list = []
        self.max_cut_number = max_cut_number    # how many video cut in one data sample
        self.use_video_frames = use_video_frames
        self.frame_stride = frame_stride  # control the sample ratio, or speed of video

        self.max_sr = max_sr       # max speed ratio
        
        # sr_label = sr - 1
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
            self.ids = db.get_cutlist()
        else:
            self.ids = train_ids if mode == 'train' else val_ids if mode == 'val' else test_ids

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
            at = read_at_from_mp4(data['front_view'], max_len=250)  # [z, 128]

            if len(at.shape) > 1 and mean:
                at = np.mean(at)

            Output_dict['at'] = at #np.expand_dims(at, 0)

        return Output_dict

    def get_features_item(self, data, Output_dict):
        pass

    def __getitem__(self, idx):

        id = self.ids[idx]
        data = db.get_cutdata(id)
        cls_name = data[db.category_key]  # or old_category_key
        cls = db.cls_mapping[cls_name]
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






