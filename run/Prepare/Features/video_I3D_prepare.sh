conda activate py310

cd feats_utils
folder="video_features"

if [! -d "$folder"]; then
  git clone https://github.com/v-iashin/video_features
fi


# need to install
# can you please run python3 -mtorch.utils.collect_env and share the output?
# pip install pydub
# OSError: libtorch_cuda_cpp.so: cannot find
# conda install cudatoolkit=11.1 -c pytorch -c conda-forge

python extract_feat.py   --video

