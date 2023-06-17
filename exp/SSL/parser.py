




import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='c3d', help='c3d/r3d/r21d')

    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--it', type=int, default=16, help='interval')
    parser.add_argument('--tl', type=int, default=3, help='tuple length')
    parser.add_argument('--max_sr', type=int, default=1, help='max clip sample rate')

    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')

    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=5, help='print frequency every batch')

    parser.add_argument('--seed', type=int, default=36, help='seed for initializing training.')

    parser.add_argument('--dataset', default='360x', type=str, help='ucf101/k400, 360x')

    parser.add_argument('--use_360', action='store_true', help='whether to use 360x dataset')
    parser.add_argument('--use_front', action='store_true', help='whether to use front view')
    parser.add_argument('--use_clip', action='store_true', help='whether to use clip view')
    parser.add_argument('--use_audio', action='store_true', help='whether to use 360x dataset')
    parser.add_argument('--use_directional_audio', action='store_true', help='whether to use 360x dataset')
    
    
    
    
    args = parser.parse_args()
    return args



