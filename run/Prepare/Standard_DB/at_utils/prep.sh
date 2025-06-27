### https://github.com/IFICL/stereocrw


# Environments
source ~/.bashrc
conda update -n base -c conda-forge conda
conda env create -f environment.yml
conda activate Stereo

#checkpoint
mkdir -p checkpoints/pretrained-models
wget -O checkpoints/pretrained-models/FreeMusic-StereoCRW-1024.pth.tar https://www.dropbox.com/s/qwepkmli4cifn84/FreeMusic-StereoCRW-1024.pth.tar?dl=1

# vis_scripts/eval_itd_in_wild.py
#In params: list_test = 'data/Youtube-ASMR/data-split/keytime/test.csv',

#checkpoint
#./scripts/evaluation/evaluation_inthewild.sh

# In-th-wild Dataset
# mkdir  dataset
# wget -O dataset/FreeMusic-StereoCRW-1024.pth.tar https://www.dropbox.com/s/be9n1jo14v7781o/in-the-wild.tar.gz?dl=1
# tar -zxvf  in-the-wild.tar.gz
# Or download from



cd Dataset/Youtube-Binaural
chmod +x download_inthewild.sh
./download_inthewild.sh

#!mv /content/stereocrw/Dataset /content/stereocrw/data

# Preprocessing the video
mkdir Dataset/DemoVideo/RawVideos/YourVideo
cd Dataset/DemoVideo
chmod +x process.sh
./process.sh 'YourVideo'

#Visualization Demo
./scripts/visualization_video.sh 'YourVideo' YOUR_SAVE_PATH




