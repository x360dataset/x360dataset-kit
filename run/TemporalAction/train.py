# python imports
import argparse
import os, time, datetime, git, sys
from pprint import pprint

# get_top_level_directory
repo = git.Repo(".", search_parent_directories=True)
top_level_directory = repo.working_tree_dir
sys.path.append(top_level_directory)


# torch imports
import torch
import torch.nn as nn
import torch.utils.data

# our code
from lib.database import database
from lib.dataset import make_dataset, make_data_loader, get_modality_usage
from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_scheduler, 
                        make_optimizer_actionformer, make_optimizer_temporalmaxer, make_optimizer_tridet,
                        fix_random_seed, ModelEma)

import warnings
warnings.filterwarnings('ignore')

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args)
    else:
        raise ValueError("Config file does not exist.")

    """1.1 Set up Right Feat Embed"""
    args = get_modality_usage(args)
    db = database(check_data=args.check)
    args.num_classes = db.temporal_mapping["counter"]
    print("=====  num_classes:", args.num_classes, "  =====")

    cfg['Ms'] = args.Ms  # Passing Modality Usage
    video_modality_num = args.Ms['front_view'] + args.Ms['binocular'] + args.Ms['panoramic']

    cfg['model']['input_dim'] = 1024 * video_modality_num + \
                                128 * video_modality_num * args.Ms['audio'] + \
                                256 * args.Ms['at']

    cfg['dataset']['feat_stride'] = 1
    cfg['dataset']['num_frames'] = 16
    cfg['dataset']['num_classes'] = args.num_classes
    cfg['model']['num_classes'] = args.num_classes

    pprint(cfg)

    # prep for output folder (based on time stamp)
    os.makedirs(cfg['output_folder'], exist_ok=True)
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], args.method, cfg_filename + "_" + args.modality + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'],  args.method, cfg_filename + "_" + args.modality + '_' + str(args.output))

    os.makedirs(ckpt_folder, exist_ok=True)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    # cfg, name, is_training, split, cfg['train_split'],
    train_dataset = make_dataset(cfg, 
        cfg['dataset_name'], "train", **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, "train", rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    if args.method == "tridet":
        make_optimizer = make_optimizer_tridet
    elif args.method == "actionformer":
        make_optimizer = make_optimizer_actionformer
    elif args.method == "temporalmaxer":
        make_optimizer = make_optimizer_temporalmaxer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(
                                        cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq
        )

        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0) and
                        (epoch > 0)
                )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument("--modality", type=str, default="10000")
    parser.add_argument("--method", type=str, default="tridet", help='tridet | actionformer | temporalmaxer')
    parser.add_argument('--check', action='store_true', help='check data')

    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=25, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    
    args = parser.parse_args()

    main(args)
