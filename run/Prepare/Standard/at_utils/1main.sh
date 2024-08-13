



source ~/.bashrc
conda activate Stereo

{

source ~/.bashrc
conda activate /bask/projects/j/jiaoj-3d-vision/Hao/anaconda/envs/Stereo



CUDA=0
Model='/bask/projects/j/jiaoj-3d-vision/Hao/360x/checkpoints/pretrained-models/FreeMusic-StereoCRW-1024.pth.tar'
csvpath='/bask/projects/j/jiaoj-3d-vision/360XProject/Data/Meta.csv'

#save_video=false

frame_rate=25
clip=0.24
patchsize=1024
patchstride=1
patchnum=512
mode='mean'
bs=12


# ------------------------------ Main -----------------------------------------#

echo 'Generating Visualization Results......'
CUDA_VISIBLE_DEVICES=$CUDA python vis_scripts/audio_tracking.py --exp=$2 \
            --setting='stereocrw_binaural' --backbone='resnet9' --batch_size=$bs \
            --num_workers=8 --max_sample=-1 --resume=$Model --patch_stride=$patchstride \
            --patch_num=$patchnum --clip_length=$clip --wav2spec --mode=$mode  --gcc_fft=$patchsize \
            --list_vis=$csvpath --no_baseline #--force 0 #--save_video $save_video

}

