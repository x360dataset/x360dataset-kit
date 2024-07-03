import argparse
import glob
from pprint import pprint

import os, time, git, sys
repo = git.Repo(".", search_parent_directories=True)
top_level_directory = repo.working_tree_dir
sys.path.append(top_level_directory)


# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from lib.database import database
from lib.dataset import make_dataset, make_data_loader, get_modality_usage
from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed
from configs import get_env


################################################################################
def main(args):
    """0. load config"""
    env = get_env()
    args = get_modality_usage(args)
    db = database(check_data=args.check)
    args.num_classes = db.temporal_mapping["counter"]
    print("=====  num_classes:", args.num_classes, "  =====")

    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args)
    else:
        raise ValueError("Config file does not exist.")
    
    #### ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    cfg['Ms'] = args.Ms  # Passing Modality Usage
    # dataset, num_classes, input_dim, feat_stride, num_frames
    video_modality_num = 1 * args.Ms['front_view'] + 1 * args.Ms['binocular'] + 1 * args.Ms['panoramic']
    cfg['model']['input_dim'] = 1024 * video_modality_num + \
                                128 * video_modality_num * args.Ms['audio'] + 256 * args.Ms['at'] 
                                # * video_modality_num

    # MAE feature length is with more 544 length
    # if "video_feat-MAE" in cfg['dataset']['panoramic_feat_file'] and args.Ms['panoramic']:
    #     cfg['model']['input_dim'] += 544
    #
    # if "video_feat-MAE" in cfg['dataset']['front_view_feat_file'] and args.Ms['front_view']:
    #     cfg['model']['input_dim'] += 544
    #
    # if "video_feat-MAE" in cfg['dataset']['binocular_feat_file'] and args.Ms['binocular']:
    #     cfg['model']['input_dim'] += 544


    cfg['dataset']['feat_stride'] = 1
    cfg['dataset']['num_frames'] = 16
    cfg['dataset']['num_classes'] = args.num_classes
    cfg['model']['num_classes'] = args.num_classes
    #### ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(cfg, 
        cfg['dataset_name'], "test", **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, "test", None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            env.json_file,
            split="test",
            tiou_thresholds=val_db_vars['tiou_thresholds'],
            post_thr=cfg['test_cfg']['post_thr']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model, -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
 ####  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    parser.add_argument("--modality", type=str, default="00000")
    parser.add_argument("--method", type=str, default="tridet", help='tridet | actionformer | temporalmaxer')
    parser.add_argument('--check', action='store_true', help='check data')
    ####  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    
    args = parser.parse_args()
    main(args)
