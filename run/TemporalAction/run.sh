#!/bin/bash
hao
cd 360x/360x_Video_Experiments/exp/TemporalAction
conda activate TemporalMaxer


# CUDA_VISIBLE_DEVICES=$0
# -------- tridet --------
echo "start training"
python train.py ./configs/tridet/360_i3d.yaml --method tridet  --output new  \
    --modality 10011

echo "start testing..."
python eval.py  ./configs/tridet/360_i3d.yaml ckpt/tridet/360_i3d_10011_new/epoch_100.pth.tar \
    --method tridet  --modality 10011 

python eval.py  ./configs/tridet/360_i3d.yaml ready_ckpt/tridet_10011.pth.tar \
    --method tridet  --modality 10011


# -------- actionformer --------
echo "start training"
python train.py ./configs/actionformer/360_i3d.yaml --method actionformer   \
    --output new  --modality 10011  #  --check

echo "start testing..."
python eval.py  ./configs/actionformer/360_i3d.yaml ckpt/actionformer/360_i3d_10011_new/epoch_150.pth.tar \
    --method actionformer  --modality 10011

python eval.py  ./configs/actionformer/360_i3d.yaml ready_ckpt/actionformer_10011.pth.tar \
    --method actionformer  --modality 10011

# OK


# -------- temporalmaxer --------
echo "start training"
python train.py ./configs/temporalmaxer/360_i3d.yaml --method temporalmaxer  \
    --output new  --modality 10011   

echo "start testing..."
python eval.py  ./configs/temporalmaxer/360_i3d.yaml \
        ckpt/temporalmaxer/360_i3d_10011_new/epoch_150.pth.tar \
        --method temporalmaxer  --modality 10011



# check would take 20mins to inspect the data integrity