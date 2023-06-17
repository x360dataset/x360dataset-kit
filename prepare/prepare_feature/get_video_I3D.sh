

folder="video_features"


if [ ! -d "$folder"]; then
  git clone https://github.com/v-iashin/video_features
fi



cd video_features
cp ../video_I3Dfeature.py .

python video_I3Dfeature.py


