import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AVE', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE, 360x')

    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])

    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)

    parser.add_argument('--use_video_frames', default=5, type=int)
    parser.add_argument('--audio_path', default='/home/hudi/data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/home/hudi/data/CREMA-D/', type=str)

    parser.add_argument('--batch_size', default=5, type=int)  # 8  24
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--learning_rate', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=0.0003, type=float, help='weight decay')

    parser.add_argument('--lr_decay_step', default=50, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=False, default=0.0, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=9999, type=int)
    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')


    parser.add_argument('--use_360', action='store_true', help='whether to use 360x dataset')
    parser.add_argument('--use_front', action='store_true', help='whether to use front view')
    parser.add_argument('--use_clip', action='store_true', help='whether to use clip view')
    
    parser.add_argument('--use_directional_audio', action='store_true', help='whether to use 360x dataset')
    parser.add_argument('--use_audio', action='store_true', help='whether to use 360x dataset')

    parser.add_argument('--aux_loss',  default=True, type=bool)
    parser.add_argument('--test_folder', default=False, type=bool)
    parser.add_argument('--use_shared_audio', default=False, type=bool)
    
    parser.add_argument('--use_stereo_audio', default=True, type=bool)
    

    return parser.parse_args()
