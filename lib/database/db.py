import sqlite3
import os
from glob import glob
from tqdm import tqdm
import librosa, cv2
import numpy as np
import pandas as pd
import multiprocessing
from .utils import db_utils


# -----------------------
# Store everything here
# -----------------------

class database(db_utils):
    def __init__(self, check_data=False):
        super(database, self).__init__(check_data)

    def get_cutlist(self):
        return list(self.cache_cut_datas.keys())

    def get_cutdata(self, id):
        return self.cache_cut_datas[id]

    def db_length(self):
        return len(self.get_idlist())

    def get_train_val_test_id(self, train_ratio=0.8, val_ratio=0.1, cutwise=True):
        if cutwise:
            return self.train_id_cutwise, self.val_id_cutwise, self.test_id_cutwise

        return self.train_ids, self.val_ids, self.test_ids

    def info_from_id(self, id):
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM {} WHERE id={}".format(self.table_name, id))
        info = cursor.fetchall()

        return info

    def col_name(self):
        cursor = self.db.cursor()
        cursor.execute("PRAGMA table_info({})".format(self.table_name))
        col_name = cursor.fetchall()
        col_name = [each[1] for each in col_name]

        return col_name

    def cls_category_mapping(self):
        pass

    def set_key_from_db(self):
        col_name = self.col_name()

        self.id_key = col_name[0]
        self.uuid_key = col_name[1]
        self.name_key = col_name[2]
        self.comment_key = col_name[3]
        self.x360_files_path_key = col_name[4]
        self.stereo_files_path_key = col_name[5]
        self.stereo_files_number_key = col_name[6]
        self.old_category_key = col_name[10]
        self.category_key = col_name[11]
        self.create_time_key = col_name[12]


    def check_single_video(self, video, resize=256):
        v = video  # data['panoramic'][cut]
        mp4name = v.split("/")[-1]
        framesnpy = v.replace(mp4name, f"frames-{resize}.npy")
        a = v.replace(".mp4", ".wav")

        try:
            try:
                d = librosa.get_duration(path=a)
            except:
                d = librosa.get_duration(filename=a)

            if not d:  # filename, path=a
                STR = "Audio sanity fails: {}".format(a)
                self.error_file.writelines(STR)
                print(STR)
                return 0
        except:
            STR = "Audio not available: {}".format(a)
            self.error_file.writelines(STR)
            print(STR)
            return 0

        if not os.path.exists(framesnpy):
            STR = "Frames NPY not available: {}".format(framesnpy)
            self.error_file.writelines(STR)
            print(STR)
            return 0

        try:
            npy = np.load(framesnpy, mmap_mode="r")
            if npy.shape[0] < 25:
                STR = "Less than one second: {}".format(framesnpy)
                self.error_file.writelines(STR)
                print(STR)
                return 0
        except:
            STR = "Frames cannot open: {}".format(framesnpy)
            self.error_file.writelines(STR)
            print(STR)
            return 0

        if not os.path.exists(v.replace(mp4name, f"video_feat.npy")):
            STR = "Video feat not available: {}".format(v.replace(mp4name, f"video_feat.npy"))
            self.error_file.writelines(STR)
            print(STR)
            return 0

        if not os.path.exists(v.replace(mp4name, f"audio_feat.npy")):
            STR = "Audio feat not available: {}".format(v.replace(mp4name, f"audio_feat.npy"))
            self.error_file.writelines(STR)
            print(STR)
            return 0

        return 1

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

    def check_data_for(self, id):
        # Read in a Video
        data = self.get_data_from_id(id, get_nested_data=True)
        c = data["category"]
        temporal_anno_data = data['temporal_anno']

        # print("temporal_anno_data:", temporal_anno_data)

        exist = 1
        max_cut_num = 6

        if c not in self.cls_mapping:
            self.cls_mapping[c] = self.cls_mapping['counter']
            self.cls_mapping[self.cls_mapping['counter']] = c
            self.cls_mapping['counter'] += 1
            exist = 0

        # Decide if it should be in train, val, or test
        # TAG = "train", Video wise

        # if exist:
        #     r = np.random.random()
        #     if r > 0.8 and r < 0.9:
        #         TAG = "val"
        #     elif r > 0.9:
        #         TAG = "test"

        max_cut_num = min([max_cut_num, len(data['panoramic']), len(data['front_view'])])
        if max_cut_num < 6:
            STR = f" Less Cut: {data['x360']}"
            self.error_file.writelines(STR)
            print(STR)

        clip_cut_maxnum = len(data['binocular']) - 1

        succ_bcut = 0

        for cut in range(max_cut_num):
            ret = self.check_single_video(data['panoramic'][cut])
            if ret == 0:
                continue
            ret = self.check_single_video(data['front_view'][cut])
            if ret == 0:
                continue

            if clip_cut_maxnum < cut:
                bcut = clip_cut_maxnum
            else:
                bcut = cut

            ret = self.check_single_video(data['binocular'][bcut])
            if ret != 0:  # succ
                succ_bcut = bcut

            v = data['front_view'][cut]
            mp4name = v.split("/")[-1]

            at = v.replace(mp4name, "at.npy")
            if not os.path.exists(at):
                STR = "at not available: {}".format(at)
                self.error_file.writelines(STR)
                print(STR)

            # temporal anno
            # temporal_anno
            temporal_anno = []
            segments = []

            fps, frame_number, duration = self.get_videoinfo(data['panoramic'][cut])

            cut_tem_start = cut * 10
            cut_tem_end = cut * 10 + 10

            annotation_list = []  # {"segment": [100, 109], "label": "eating", "label_id": 1}

            for anno in temporal_anno_data:
                label = anno["label"]

                anno_start = anno["start"]
                anno_end = anno["end"]

                # Out of range
                if anno_start > cut_tem_end:
                    continue
                if anno_end < cut_tem_start:
                    continue

                real_start = np.maximum(anno_start, cut_tem_start) % 10
                real_end = np.minimum(anno_end, cut_tem_end) % 10
                if anno_end > cut_tem_end:
                    real_end = 10

                if real_end - real_start < 0.5:
                    continue

                if real_end < 1e-6:
                    continue

                if label not in self.temporal_mapping:
                    self.temporal_mapping[label] = self.temporal_mapping['counter']
                    self.temporal_mapping[self.temporal_mapping['counter']] = label
                    self.temporal_mapping['counter'] += 1

                temporal_anno.append(self.temporal_mapping[label])  # change to list
                segments.append([real_start, real_end])

                annotation_list.append({"segment": [real_start, real_end],  # * fps
                                        "label": label,
                                        "label_id": self.temporal_mapping[label]})

                # , "start": segment[0], "end": segment[1]}

            segments = np.asarray(segments, dtype=np.float16)
            cache_data = data.copy()

            cache_data['fps'] = fps
            cache_data['frame_number'] = frame_number
            cache_data['duration'] = duration

            cache_data['temporal_label'] = temporal_anno
            cache_data['segments'] = segments

            cache_data['panoramic'] = cache_data['panoramic'][cut]
            cache_data['front_view'] = cache_data['front_view'][cut]
            cache_data['binocular'] = cache_data['binocular'][succ_bcut]
            cache_data['monocular'] = cache_data['monocular'][succ_bcut]


            if cut == 4:
                TAG = "val"
            elif cut == 5:
                TAG = "test"
            else:
                TAG = "train"

            self.temporal_label_dict["database"][self.cacheid] = \
                {"duration": duration, "resolution": [256, 512],
                  "subset": TAG, "annotations": annotation_list}

            self.cache_cut_datas[self.cacheid] = cache_data


            if cut > 3 and cut < 5:
                self.val_id_cutwise.append(self.cacheid)
            elif cut > 4:
                self.test_id_cutwise.append(self.cacheid)
                # print("len of test:", len(self.test_id_cutwise))
            else:
                self.train_id_cutwise.append(self.cacheid)

            if TAG == "train":
                self.train_ids.append(self.cacheid)
            elif TAG == "val":
                self.val_ids.append(self.cacheid)
            elif TAG == "test":
                self.test_ids.append(self.cacheid)

            self.cacheid += 1
            
    def check_data_qualify(self, max_cut_num=6):
        print("=== Checking data qualify ===")
        self.cacheid = 0

        for id in tqdm(self.idlist):
            self.check_data_for(id)

        ids = {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids,
               "train_cut": self.train_id_cutwise, "val_cut": self.val_id_cutwise,
               "test_cut": self.test_id_cutwise}

        np.save(self.cache_root, self.cache_cut_datas)
        np.save(self.cls_mapping_npy, self.cls_mapping)
        np.save(self.temporal_mapping_npy, self.temporal_mapping)
        np.save(self.ids_npy, ids)
        np.save(self.temporal_label_json, self.temporal_label_dict)
        print("checked LENGTH of self.cache_cut_datas:", self.cache_cut_datas.__len__())

    def load_cache(self):
        self.cache_cut_datas = np.load(self.cache_root, allow_pickle=True).item()
        self.cls_mapping = np.load(self.cls_mapping_npy, allow_pickle=True).item()
        self.temporal_mapping = np.load(self.temporal_mapping_npy, allow_pickle=True).item()

        ids = np.load(self.ids_npy, allow_pickle=True).item()
        self.train_ids = ids["train"]
        self.val_ids = ids["val"]
        self.test_ids = ids["test"]
        self.train_id_cutwise = ids["train_cut"]
        self.val_id_cutwise = ids["val_cut"]
        self.test_id_cutwise = ids["test_cut"]

        print("LENGTH of self.cache_cut_datas:", self.cache_cut_datas.__len__())

    def get_data_from_id(self, id, get_nested_data=False):

        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM {} WHERE id={}".format(self.table_name, id))
        data = cursor.fetchall()[0]  # output is a tuple

        suffix = "_trim"

        outdata_dict = {}
        outdata_dict["id"] = data[0]
        outdata_dict["uuid"] = data[1]

        anno = os.path.join(self.action_anno_root, data[1], "*.json")
        anno = list(glob(anno))
        if len(anno) < 1:
            outdata_dict['temporal_anno'] = []  # Vacant
        else:
            anno = anno[0]
            metadata = pd.read_json(anno)['metadata']
            anno_processed = []
            # print("keys:", metadata.keys())

            for idx, key in enumerate(metadata.keys()):

                try:
                    label = metadata[key]['av']['1']
                except:
                    # print("anno file:", anno)
                    # print("file:", pd.read_json(anno)['metadata'] )
                    # print("failed key:", key)
                    continue

                segment = metadata[key]['z']

                try:
                    s = {"label": label, "start": segment[0], "end": segment[1]}
                except:
                    print("anno file:", anno)
                    print("failed segments:", segment)
                    continue

                # print("anno s:", s)
                anno_processed.append(s)

            # print("anno_processed=",anno_processed)

            outdata_dict['temporal_anno'] = anno_processed  # List of

            # "metadata":{
            # "1_opxzNF0y":{"vid":"1","flg":0,"z":[7.29542,35.04542],"xy":[],"av":{"1":"operating phone"}},
            # "1_0Fc6K9IU":{"vid":"1","flg":0,"z":[37.17,44.79542],"xy":[],"av":{"1":"walking"}},

        outdata_dict["name"] = data[2]
        outdata_dict["comment"] = data[3]

        outdata_dict["x360_original"] = data[4]
        outdata_dict["stereo_original"] = data[5]
        # outdata_dict["stereo_number"] = data[6]

        outdata_dict["old_category"] = data[10]
        outdata_dict["category"] = data[11]

        outdata_dict['x360'] = data[4].replace(self.original_datafolder, self.trim_datafolder) + suffix
        outdata_dict['stereo'] = data[5].replace(self.original_datafolder, self.trim_datafolder) + suffix
        outdata_dict["time"] = data[8]

        if get_nested_data:
            outdata_dict['panoramic'] = \
                sorted(list(glob(os.path.join(outdata_dict['x360'], "360_panoramic" + suffix, "*", "*.mp4"))))

            outdata_dict['front_view'] = \
                sorted(list(glob(os.path.join(outdata_dict['x360'], "front_view" + suffix, "*", "*.mp4"))))

            outdata_dict["binocular"] = \
                sorted(list(glob(os.path.join(outdata_dict['stereo'], "binocular" + suffix, "*", "*", "*.mp4"))))

            outdata_dict["monocular"] = \
                sorted(list(glob(os.path.join(outdata_dict['stereo'], "monocular" + suffix, "*", "*", "*.mp4"))))

            # outdata_dict['at'] = \
            #     list(glob(os.path.join(outdata_dict['x360'], "front_view" + suffix, "*", "at.npy")))

        # print("DEBUG data:", outdata_dict)

        return outdata_dict

    def get_videos_list(self, istrim=False):
        output_dict = {}
        output_dict['panoramic'] = []
        output_dict['binocular'] = []
        output_dict['monocular'] = []
        output_dict['front_view'] = []

        for id in self.get_idlist():
            if not istrim:
                data = self.get_data_from_id(id)
                # 360/360_panoramic.mp4
                # Stereo/binocular/clip*/binocular.mp4

                output_dict['panoramic'].append(
                    os.path.join(data['x360_original'], "360_panoramic.mp4")
                )
                output_dict['front_view'].append(
                    os.path.join(data['x360_original'], "front_view.mp4")
                )
                output_dict["binocular"].extend(
                    list(glob(os.path.join(data['stereo_original'], "*", "binocular.mp4")))
                )
                output_dict["monocular"].extend(
                    list(glob(os.path.join(data['stereo_original'], "*", "monocular.mp4")))
                )

            else:
                data = self.get_data_from_id(id)
                suffix = "_trim"
                # 360_trim/360_panoramic_trim/cut0_0s_to_10s/cut0_0s_to_10s.mp4
                # Stereo_trim/binocular_trim/clip*/cut0_0s_to_10s/cut0_0s_to_10s.mp4

                path = os.path.join(data['x360'], "360_panoramic" + suffix, "*", "*.mp4")

                output_dict['panoramic'].extend(
                    list(glob(path))
                )
                output_dict['front_view'].extend(
                    list(glob(os.path.join(data['x360'], "front_view" + suffix, "*",
                                           "*.mp4")))
                )

                # print("BINOCULAR PATH:", os.path.join(data['stereo'], "binocular"+suffix))

                output_dict["binocular"].extend(
                    list(glob(os.path.join(data['stereo'], "binocular" + suffix,
                                           "*", "*", "*.mp4")))
                )
                output_dict["monocular"].extend(
                    list(glob(os.path.join(data['stereo'], "monocular" + suffix, "*",
                                           "*", "*.mp4")))
                )

        return output_dict

    def get_video_info(self, mp4path):
        cv2cap = cv2.VideoCapture(mp4path)
        fps = cv2cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cv2cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        return fps, frame_count, duration

    def get_statistics(self):
        TOTAL_time = {}
        Video_number = {}
        Frame_number = {}
        clips_number = {}

        video_list = self.get_videos_list(istrim=False)

        for item in video_list:
            l = video_list[item]
            TOTAL_time[item] = 0
            Video_number[item] = 0
            Frame_number[item] = 0
            clips_number[item] = len(l)

            for i in tqdm(l):
                Video_number[item] += 1
                fps, frame_count, duration = self.get_video_info(i)
                Frame_number[item] += frame_count
                TOTAL_time[item] += duration

            print(" =====  ", item, "  =====")
            print("TOTAL_time:", TOTAL_time[item])
            print("Video_number:", Video_number[item])
            print("Frame_number:", Frame_number[item])
            print("clips_number:", clips_number[item])

        print(" =====  Statistics  =====")
        print("TOTAL_time:", TOTAL_time)
        print("Video_number:", Video_number)
        print("Frame_number:", Frame_number)
        print("clips_number:", clips_number)


    def get_at_time_span(self):
        # video_list = self.get_videos_list(istrim=True)

        i1, i2, i3 = self.get_train_val_test_id()

        id_list = i1 + i2 + i3

        out_at_list = []
        in_at_list = []
        at_all_list_out = []

        at_all_list_in = []

        def read_at_from_mp4(mp4file, max_len=250):
            name = mp4file.split("/")[-1]
            at = np.load(mp4file.replace(name, "at.npy"))[:max_len]
            if len(at) < 255:
                at = np.pad(at, (0, 255 - len(at)), 'constant', constant_values=(0, 0))
            return at

        print("Length:", len(id_list))
        for idx in tqdm(id_list):
            data = db.get_cutdata(idx)
            cato = data['category']

            # print("data['front_view']:", data['front_view'])
            at_ = read_at_from_mp4(data['front_view'])
            # print("at:", len(at_list))

            if cato in self.outdoor:
                out_at_list.append([0, np.mean(at_)])
                at_all_list_out.extend(at_.tolist())
            elif cato in self.indoor:
                in_at_list.append([0, np.mean(at_)])
                at_all_list_in.extend(at_.tolist())
            else:
                if np.random.random() < 0.4:
                    out_at_list.append([0, np.mean(at_)])
                    at_all_list_out.extend(at_.tolist())
                else:
                    in_at_list.append([0, np.mean(at_)])
                    at_all_list_in.extend(at_.tolist())

        at_list = {"outdoor": out_at_list, "indoor": in_at_list}

        np.save("out_at_statistics.npy", out_at_list)
        np.save("in_at_statistics.npy", in_at_list)
        np.save("all_at_statistics.npy", at_all_list_out)
        np.save("all_at_statistics_in.npy", at_all_list_in)

        print("Statistics saved to at_statistics.npy")

        out_at_list = np.load("out_at_statistics.npy")
        in_at_list = np.load("in_at_statistics.npy")
        # out_at_list = np.array(out_at_list)

        bin = len(out_at_list)
        import matplotlib.pyplot as plt
        plt.plot(list(range(bin)), out_at_list, alpha=0.5, label='positive')

        in_at_list = np.array(in_at_list)
        plt.plot(np.array(list(range(len(in_at_list)))) + bin, in_at_list, alpha=0.5, label='negative')

        plt.savefig("at_statistics.png")


if __name__ == "__main__":
    db = database()
    print("db col name:", db.col_name())
    print("db length:", (len(db.get_idlist())))
    print(db.info_from_id(1))
    print("db list:", db.get_idlist())

    print("cls mapping:", db.cls_mapping)

    print("temporal mapping:", db.temporal_mapping)

    db.get_at_time_span()


    # "database": {"Pub_10_SpottedDog_20220101_000000_100s_to_110s":
    # {"duration": 9, "resolution": [2880.0, 5760.0], "subset": "training",
    # "annotations":
    # [{"segment": [100, 109], "label": "eating", "label_id": 1},
    # {"segment": [100.694, 109], "label": "walking", "label_id": 2},
    # {"segment": [100, 109], "label": "speaking", "label_id": 3}]},
    # "Kitchen_07_Dishes_20220101_000000_510s_to_520s": {"duration": 9, "resolution": [2880.0, 5760.0], "subset": "training", "annotations": []},
