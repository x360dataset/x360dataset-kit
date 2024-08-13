import cv2
import os
from glob import glob
from tqdm import tqdm


def get_save_folder(each_video, mp4name):
    frame_folder = each_video.replace(mp4name, "frames")

    return frame_folder



class videoReader(object):
    def __init__(self, video_path,
                 frame_interval=1, frame_kept_per_second=1):

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
            # os.system(f"rm -r {frame_folder}")
            # os.remove(self.video_path)
            self.folder_status = 0


    def video2frame(self, frame_save_path, force, verbose=False):
        if self.folder_status == 0:
            return

        self.frame_save_path = frame_save_path

        count = 0
        time = 0
        frame_interval = 1   #int(self.fps/self.frame_kept_per_second)   # int > 1
        # print("handling frame_interval:", frame_interval)

        if not force and self.video_frames/self.fps*self.frame_kept_per_second/frame_interval - 5 \
                      <= len(list(glob(self.frame_save_path + "*.jpg"))):
            # if verbose:
            #     print("Already processed FRAME: {}".format(self.video_path), "\n")
            return

        while(count < self.video_frames):
            ret, image = self.vid.read()
            time += frame_interval

            if not ret and count > 0:
                print(f"break with status {ret}")
                break

            if count % self.fps == 0:  # every second save_name top n frame
                frame_id = 0           # Get top frame

            # new frame_id
            # and frame_id%frame_interval == 0
            if count/self.fps*self.frame_kept_per_second > frame_interval * frame_id:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                if not os.path.exists(save_name):
                    cv2.imencode('.jpg', image)[1].tofile(save_name)
                frame_id += 1
            count += 1

def is_dir_empty(dir_path):
    return not os.listdir(dir_path)




class mp4list_to_frames():
    def __init__(self,
        videolist = [], videoname="", frame_interval=1, frame_kept_per_second=1):

        self.frame_kept_per_second = frame_kept_per_second
        self.frame_interval = frame_interval
        self.videos_list = videolist
        self.videoname = videoname

    def extractImage(self, force=False, DEBUG=False, strict_check=False, verbose=False):

        PROCESSED = 0
        for i, each_video in tqdm(enumerate(self.videos_list)):
            mp4name = each_video.split("/")[-1]
            if mp4name.endswith(".mp4"):

                # TODO: the save folder
                frame_folder = get_save_folder(each_video, mp4name)


                if (not strict_check and os.path.exists(frame_folder)) or \
                        len(list(glob(os.path.join(frame_folder, "*.jpg"))))  > 25:
                    # print("Already processed: {}".format(each_video))
                    continue

                if verbose:
                    print("source: ", each_video, "\n  -->target_folder: ", frame_folder)

                os.makedirs(frame_folder, exist_ok=True)

                # try:
                self.videoReader = videoReader(video_path=each_video,
                                               frame_interval=self.frame_interval,
                                               frame_kept_per_second=self.frame_kept_per_second)

                self.videoReader.video2frame(frame_save_path=frame_folder, force=force, verbose=verbose)

