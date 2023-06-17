

#!/bin/bash.sh
frame_rate=1

echo "Name of the video folder: $1"
echo "Frame rate: $frame_rate"


python create-csv.py --dataset_name=$1 --type='' --data_split='1:0:0' --unshuffle

