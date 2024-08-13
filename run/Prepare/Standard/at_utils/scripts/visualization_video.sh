#!/bin/bash.sh
{

CUDA=0
Model='../../checkpoints/pretrained-models/FreeMusic-StereoCRW-1024.pth.tar'

clip=0.24
patchsize=3840
patchstride=1
patchnum=512
mode='mean'

# ------------------------------ Main -----------------------------------------#

csvpath='/bask/projects/j/jiaoj-3d-vision/360XProject/Data/Meta.csv'

source ~/.bashrc
conda activate Stereo

echo 'Generating Visualization Results......'
CUDA_VISIBLE_DEVICES=$CUDA python vis_scripts/vis_video_itd.py --exp=$2 \
            --setting='stereocrw_binaural' --backbone='resnet9' --batch_size=2 \
            --num_workers=8 --max_sample=-1 --resume=$Model --patch_stride=$patchstride \
            --patch_num=$patchnum --clip_length=$clip --wav2spec --mode=$mode  --gcc_fft=$patchsize \
            --list_vis=$csvpath --no_baseline

}