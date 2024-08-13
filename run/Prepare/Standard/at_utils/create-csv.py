import os, glob, random, json, argparse, csv, cv2
import numpy as np
from tqdm import tqdm
import soundfile as sf


parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--exam', default=False, action='store_true')
parser.add_argument('--type', default='', type=str)
parser.add_argument('--data_split', default='8:1:1', type=str)
parser.add_argument('--split_by_video', default=False, action='store_true')
parser.add_argument('--unshuffle', default=False, action='store_true')

random.seed(1234)
DEBUG = False


# root = "/bask/projects/j/jiaoj-3d-vision/360XProject/Data"
# video_list = glob.glob(os.path.join(root, "*", "*", "*", "360/front_view_cut/*.mp4"))
# Check Data

newroot = "/bask/projects/j/jiaoj-3d-vision/Hao/360Data_2023Oct"
video_list = glob.glob(os.path.join(newroot, "*/360/front_view_cut/*.mp4"))



video_list = [i for i in video_list if "*" not in i]
video_list.sort()
print(f"processing {len(video_list)} videos...")


def get_frameinfo_fromvideo(video_path):
    vid = cv2.VideoCapture(video_path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    video_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    # video_len = int(self.video_frames/self.fps)
    return fps, video_frames


def write_csv(data_list, filepath):
    # import pdb; pdb.set_trace()
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ["path"]  # "train/ test/ val  "#list(data_list[0].keys())
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for info in data_list:
            writer.writerow(info)

    print('{} items saved to {}.'.format(len(data_list), filepath))


def create_list_for_video(args, name, video_list):
    # import pdb; pdb.set_trace()
    totol_pass = 0

    sample_list = []
    if args.split_by_video:
        clip_list = []
        for video in video_list:
            temp = glob.glob(f'{video}/*')
            temp.sort()
            clip_list += temp
        new_video_list = clip_list
    else:
        new_video_list = video_list

    for video in tqdm(new_video_list):
        if DEBUG: print("video: ", video)
        path = video
        video_name = path.split('/')[-1]

        sp = video_name.split("_")[1]
        # print("sp:", sp)
        filter_cutnumber = 6 # 6

        if not video.endswith('.mp4') or int(sp) >= filter_cutnumber:      # or not video_name.startswith("cut_"):
            totol_pass += 1
            continue

        fps, frame_num = get_frameinfo_fromvideo(video)
        if DEBUG: print("fps: ", fps)
        if DEBUG: print("frame_num: ", frame_num)

        video = video.rstrip(video_name).rstrip("/")
        try:
            audio_path = os.path.join(video, 'audios', video_name.replace(".mp4", '.wav'))
            audio, audio_rate = sf.read(audio_path, dtype='int16')
        except:
            print("audio_path broken: ", audio_path)
            totol_pass += 1
            continue

        
        if fps < 1:
            cond_3 = True
        else:
            cond_3 = np.abs(audio.shape[0] / audio_rate - frame_num / fps) > 0.1

        if cond_3:
            continue

        if name == 'train':
            sample_list.append({'path': path})

    print("===totol_pass: ", totol_pass)
    return sample_list




def main(args):
    test_num = -1 if not DEBUG else 3
    data_list = video_list[:test_num]
    
    sample_list = create_list_for_video(args, 'train', data_list)
    print("output sample_list length:", len(sample_list))

    name = 'Meta'
    output_dir = "/bask/projects/j/jiaoj-3d-vision/360XProject/Data"
    csv_name = f'{output_dir}/{name}.csv'

    write_csv(sample_list, csv_name)



if __name__ == "__main__":
    args = parser.parse_args()
    
    main(args)
    