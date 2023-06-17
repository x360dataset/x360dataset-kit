import pandas as pd
import cv2
import os
import pdb
from glob import glob

class videoReader(object):
    def __init__(self, video_path, cut_frame_folder,frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        self.folder_status = 1
        #pdb.set_trace()
        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        try:
            self.video_len = int(self.video_frames/self.fps)
            print(f"video with frames {self.video_frames}, fps:", self.fps, " and fps keeped:", self.frame_kept_per_second)

        except:
            print("video broken: ", self.video_path, "with fps:", self.fps)
            os.system(f"rm -r {cut_frame_folder}")
            os.remove(self.video_path)
            self.folder_status = 0

        # print("self.video_fps: ", self.fps)

    def video2frame(self, frame_save_path):
        if self.folder_status == 0:
            return

        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0

        while success:
            count +=1
            if count % self.frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count/self.fps), count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()

    def video2frame_timer(self, frame_save_path):
        if self.folder_status == 0:
            return

        frame_interval = self.frame_kept_per_second/self.fps
        timer = 0
        count = 0
        self.frame_save_path = frame_save_path
        print("handling frame_interval:", frame_interval)

        # head
        ret, image = self.vid.read()

        while (count < self.video_frames):
            ret, image = self.vid.read()
            if not ret:
                count += 1
                continue
                # break
            print("success")
            timer += frame_interval
            if frame_interval > 1:
                frame_interval -= 1
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)

            count += 1


    def video2frame_update(self, frame_save_path):
        if self.folder_status == 0:
            return

        self.frame_save_path = frame_save_path

        count = 0
        time = 0
        frame_interval = 1   #int(self.fps/self.frame_kept_per_second)   # int > 1
        print("handling frame_interval:", frame_interval)

        while(count < self.video_frames):
            ret, image = self.vid.read()
            time += frame_interval
            if not ret and count > 0:
                print(f"break with status {ret}")
                break

            if count % self.fps == 0:  # every second save_name top n frame
                frame_id = 0

            if frame_id<frame_interval*self.frame_kept_per_second and frame_id%frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)

            frame_id += 1
            count += 1

def is_dir_empty(dir_path):
    return not os.listdir(dir_path)

class process_dataset(object):
    def __init__(self,
        videolist = [], videoname="", frame_interval=1, frame_kept_per_second=1):

        self.frame_kept_per_second = frame_kept_per_second
        self.frame_interval = frame_interval
        self.videos_list = videolist
        self.videoname = videoname

    def extractImage(self, force=False):

        for i, each_video in enumerate(self.videos_list):

            mp4name = each_video.split("/")[-1]

            if mp4name.endswith(".mp4") and mp4name.startswith("cut_"):
                cut_frame_folder = os.path.join(each_video.rstrip(mp4name),
                                            'all_frames', mp4name.rstrip(".mp4"))



                if not os.path.exists(cut_frame_folder) or force or is_dir_empty(cut_frame_folder):
                    if is_dir_empty(cut_frame_folder):
                        print("empty:", cut_frame_folder)

                    print("Processing: {}".format(each_video))
                    os.makedirs(cut_frame_folder, exist_ok=True)
                    # try:
                    self.videoReader = videoReader(video_path=each_video,
                                                   cut_frame_folder = cut_frame_folder,
                                                   frame_interval=self.frame_interval,
                                                   frame_kept_per_second=self.frame_kept_per_second)

                    self.videoReader.video2frame_update(frame_save_path=cut_frame_folder)

                    # except:
                    #     print('Fail @ {}'.format(each_video[:-1]))
                else:
                    pass
                    # print("Already processed: {}".format(each_video))



root = '../Data'
folder =  glob(os.path.join(root, "Inside", "*", "*")) + \
          glob(os.path.join(root, "Outside", "*", "*"))

from glob import glob
from tqdm import tqdm

for videoid in tqdm(folder):
    _360_list = glob(os.path.join(videoid, "360/360_panoramic_cut/*"))
    _360_front_list = glob(os.path.join(videoid, "360/front_view_cut/*"))
    _Snapchat_list = glob(os.path.join(videoid, "Snapchat/*/binocular_cut/*"))


    x360_fps = 25  # not use
    fps = 25  # not use

    # D = process_dataset(videolist=_360_list,
    #                     frame_interval=1, frame_kept_per_second=x360_fps)
    # D.extractImage()

    D = process_dataset(videolist=_360_front_list,
                        frame_interval=1, frame_kept_per_second=x360_fps)
    D.extractImage()

    # 60 fps
    # D = process_dataset(videolist=_Snapchat_list,
    #                     frame_interval=1, frame_kept_per_second=fps)
    # D.extractImage()
