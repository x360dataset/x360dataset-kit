import git, sys
# get_top_level_directory
repo = git.Repo(".", search_parent_directories=True)
top_level_directory = repo.working_tree_dir
sys.path.append(top_level_directory)
from lib.database import database

import argparse, random
from feats_utils import get_VideoFeature
from feats_utils import get_AudioFeature, get_VideoFeature_MAE


db = database(check_data=False)

parser = argparse.ArgumentParser()
parser.add_argument('--audio', action='store_true', default=False)
parser.add_argument('--video', action='store_true', default=False)
parser.add_argument('--MAE', action='store_true', default=False)

parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--force', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--reverse', action='store_false')


args = parser.parse_args()

extract_audio_feats = args.audio
extract_video_feats = args.video
MAE = args.MAE
DEBUG = args.debug
force = args.force
verbose = args.verbose
reverse = args.reverse


if extract_audio_feats:
    # It packs the data into a frame single file
    video_list = db.get_videos_list(istrim=True)

    print("=== Handling extraction task: Extract Audio Feats ===")
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
        pretrain_path = None
        get_AudioFeature(l, force=force, pretrain_path=pretrain_path)


if extract_video_feats:
    # It packs the data into a frame single file
    video_list = db.get_videos_list(istrim=True)

    print("=== Handling extraction task: Extract Video Feats ===")
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
        pretrain_path = None
        get_VideoFeature(l, force=force, pretrain_path=pretrain_path)



if MAE:
    # It packs the data into a frame single file
    video_list = db.get_videos_list(istrim=True)

    print("=== Handling extraction task: Extract Video Feats ===")
    print("=== stat of video_list ===")
    print(f"panoramic: {len(video_list['panoramic'])}")
    print(f"front_view: {len(video_list['front_view'])}")
    print(f"stereo binocular: {len(video_list['binocular'])}")
    print(f"stereo monocular: {len(video_list['monocular'])}")


    for item in video_list:
        print(f"==== process {item} videos: {len(video_list[item])}")
        if item == "monocular":
            continue
        if item == "binocular":
            continue

        l = video_list[item]

        random.shuffle(l)
        print("HANDLEING length:", len(l))
        pretrain_path = None
        get_VideoFeature_MAE(l, force=force, pretrain_path=pretrain_path)
