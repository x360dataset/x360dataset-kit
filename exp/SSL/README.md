# Video Self-supervised Methods

This repo is mainly based on [video-pace](https://github.com/laura-wang/video-pace) and [video-clip-order](https://github.com/xudejing/video-clip-order-prediction).

## Main Dependencies
+ Ubuntu 16.04
+ CUDA Version: 11.1
+ PyTorch 1.8.1
+ torchvision 0.9.1
+ python 3.7.6

### Data Preparation
Download the Original [Dataset](https://x360dataset.github.io/).

### Train and Test

You can self-supervised train the model and test it by simply running

**360 panoramic video**

```bash scripts/360.sh ```

**front cut video**

```bash scripts/front.sh ```

**egocentric stereo video**

```bash scripts/clip.sh ```


#### Acknowledgement

If you use this repo, please cite the original works: [video-pace](https://github.com/laura-wang/video-pace) and [video-clip-order](https://github.com/xudejing/video-clip-order-prediction).

