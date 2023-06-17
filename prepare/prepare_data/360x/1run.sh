

source ~/.bashrc
#conda activate Stereo
conda activate py37

# Step1
python ./process_data/360x/mp4_to_wav.py

# Step2
python ./process_data/360x/mp4_to_frames.py

# Step3
python ./process_data/360x/generate_csv.py




# Download
# cd ../Data
