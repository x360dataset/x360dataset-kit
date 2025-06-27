import argparse
import os
import shutil


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process 360 subfolders and rename files.')
    parser.add_argument('--root_path', type=str, help='Root directory containing folders to process')
    args = parser.parse_args()

    all_case = os.listdir(args.root_path)
    print(all_case)
    print("len = ", len(all_case))