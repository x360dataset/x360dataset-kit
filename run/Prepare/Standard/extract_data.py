import git, sys, os
# get_top_level_directory
repo = git.Repo(".", search_parent_directories=True)
top_level_directory = repo.working_tree_dir
sys.path.append(top_level_directory)
from lib.database import database

from utils import generate_wav_from_mp4list, trim_video_from_mp4list
from utils import mp4list_to_frames, packs_frame_folder
import argparse
import random
db = database(check_data=False)


parser = argparse.ArgumentParser()

parser.add_argument('--trim', action='store_true', default=False)  # cut video into cuts
parser.add_argument('--audio', action='store_true', default=False)
parser.add_argument('--frame', action='store_true', default=False) # extract frames
parser.add_argument("--at", action='store_true', default=False, help="audio tracking")
parser.add_argument('--pack', action='store_true')                 # pack frames


parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--force', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--reverse', action='store_false')

parser.add_argument('--save_video', type=bool, default=False, required=False)

args = parser.parse_args()

trim_video = args.trim
extract_wav = args.audio
extract_frame = args.frame
get_at = args.at
pack = args.pack

DEBUG = args.debug
force = args.force
verbose = args.verbose
reverse = args.reverse


pack_frame = args.pack





if trim_video:
    video_list = db.get_videos_list(istrim=False)

    print("=== Handling extraction task: trim_video ===")
    print("=== stat of video_list ===")
    print(f"panoramic: {len(video_list['panoramic'])}")
    print(f"front_view: {len(video_list['front_view'])}")
    print(f"stereo binocular: {len(video_list['binocular'])}")
    print(f"stereo monocular: {len(video_list['monocular'])}")


    for item in video_list:
        if item == "panoramic" or item == "front_view":
            continue

        l = video_list[item]
        # if reverse:
        random.shuffle(l)

        print(f"==== process {item} videos: {len(l)}")
        # print("l:", l)

        trim_video_from_mp4list(l,
                                videoname=item,
                                force=force,
                                DEBUG=DEBUG,
                                verbose=verbose)
        # break


if extract_wav:
    video_list = db.get_videos_list(istrim=True)

    print("=== Handling extraction task: extract_wav ===")
    print("=== stat of video_list ===")
    print(f"panoramic: {len(video_list['panoramic'])}")
    print(f"front_view: {len(video_list['front_view'])}")
    print(f"stereo binocular: {len(video_list['binocular'])}")
    print(f"stereo monocular: {len(video_list['monocular'])}")

    for item in video_list:
        print(f"==== process {item} videos: {len(video_list[item])}")
        if item == "panoramic" or item == "front_view":
            continue

        l = video_list[item]

        random.shuffle(l)
        print("HANDLEING length:", len(l))

        generate_wav_from_mp4list(l, force=force, DEBUG=DEBUG, verbose=verbose)


if extract_frame:
    video_list = db.get_videos_list(istrim=True)

    print("=== Handling extraction task: extract_frame ===")
    print("=== stat of video_list ===")
    print(f"panoramic: {len(video_list['panoramic'])}")
    print(f"front_view: {len(video_list['front_view'])}")
    print(f"stereo binocular: {len(video_list['binocular'])}")
    print(f"stereo monocular: {len(video_list['monocular'])}")


    for item in video_list:
        print(f"==== process {item} videos: {len(video_list[item])}")
        if item == "panoramic" or item == "front_view" or item == "monocular":
            continue

        l = video_list[item]

        random.shuffle(l)

        frame_kept_per_second = 25  # 25
        frame_interval = 1
        D = mp4list_to_frames(l, videoname=item,
                              frame_interval=frame_interval,
                              frame_kept_per_second = frame_kept_per_second)

        strict_check = True

        D.extractImage(force = force, DEBUG = DEBUG,
                       strict_check=strict_check, verbose=verbose)


if pack_frame:
    # It packs the data into a frame single file
    video_list = db.get_videos_list(istrim=True)

    print("=== Handling extraction task: packing frames ===")
    print("=== stat of video_list ===")
    print(f"panoramic: {len(video_list['panoramic'])}")
    print(f"front_view: {len(video_list['front_view'])}")
    print(f"stereo binocular: {len(video_list['binocular'])}")
    print(f"stereo monocular: {len(video_list['monocular'])}")

    for item in video_list:
        print(f"==== process {item} videos: {len(video_list[item])}")

        l = video_list[item]

        random.shuffle(l)
        print("HANDLEING length:", len(l))

        packs_frame_folder(l)


if get_at:
    os.system("python at_utils/main.sh")

    # apply_at_to_datalist(db, force=force, DEBUG=DEBUG, save_video=args.save_video)