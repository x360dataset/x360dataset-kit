




source ~/.bashrc
#conda activate Stereo
conda activate py37



# Step 1
python process_data/360x/trim/trim_360_videos.py

# Step 2
python process_data/360x/trim/trim_mp4_to_wav.py

# Step 3
python process_data/360x/trim/trim_mp4_to_frames.py
