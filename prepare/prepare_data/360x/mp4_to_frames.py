import pandas as pd
import cv2
import os
import pdb
from glob import glob

class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        #pdb.set_trace()
        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames/self.fps)


    def video2frame(self, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count +=1
            if count % self.frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count/self.fps), count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()


    def video2frame_update(self, frame_save_path):
        self.frame_save_path = frame_save_path

        count = 0
        frame_interval = int(self.fps/self.frame_kept_per_second)
        while(count < self.video_frames):
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id<frame_interval*self.frame_kept_per_second and frame_id%frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)

            frame_id += 1
            count += 1

class process_dataset(object):
    def __init__(self,
        videolist = [], videoname="", frame_interval=1, frame_kept_per_second=1):

        self.frame_kept_per_second = frame_kept_per_second
        self.frame_interval = frame_interval
        self.videos_list = videolist
        self.videoname = videoname
        print(f"processing {len(self.videos_list)} videos...")

    def extractImage(self):

        for i, each_video in enumerate(self.videos_list):
            if i % 1 == 0:
                print('*******************************************')
                print('Processing: {}/{}'.format(i, len(self.videos_list)))
                print('*******************************************')

            mp4name = each_video.split("/")[-1]
            if mp4name == self.videoname:
                print("Processing: {}".format(each_video))
                frame_folder = self.videoname.rstrip(".mp4") + '-frame' + '-{:02d}-FPS'.format(self.frame_kept_per_second)
                frame_folder = each_video.replace(self.videoname, frame_folder)

                if not os.path.exists(frame_folder):
                    os.mkdir(frame_folder)

                    try:
                        self.videoReader = videoReader(video_path=each_video,
                                                       frame_interval=self.frame_interval,
                                                       frame_kept_per_second=self.frame_kept_per_second)

                        self.videoReader.video2frame_update(frame_save_path=frame_folder)
                    except:
                        print('Fail @ {}'.format(each_video[:-1]))
                else:
                    print("Already processed: {}".format(each_video))



root = '../Data'
folder =  glob(os.path.join(root, "Inside", "*", "*")) + \
          glob(os.path.join(root, "Outside", "*", "*"))

for videoid in folder:
    _360_list = glob(os.path.join(videoid, "360", "*"))
    _Snapchat_list = glob(os.path.join(videoid, "Snapchat/*/*"))

    # Important Params: frame_interval=1, frame_kept_per_second=1
    D = process_dataset(videolist=_360_list, videoname="360_panoramic.mp4",
                        frame_interval=1, frame_kept_per_second=1)
    D.extractImage()

    D = process_dataset(videolist=_Snapchat_list, videoname="binocular.mp4",
                        frame_interval=1, frame_kept_per_second=1)
    D.extractImage()
