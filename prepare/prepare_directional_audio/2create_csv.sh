# activate conda env
# framerate = 10?

source ~/.bashrc
conda activate Stereo


# mk dataset
cd DemoVideo
chmod +x create_csv.sh
./create_csv.sh 'YourVideo'

