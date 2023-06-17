from parser import *
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb

from dataset.x360dataset import x360Dataset, x360DatasetTest
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init
from functions import *

import wandb


project = "360x"
exp_label = "at"

def main():
    args = get_arguments()

    # name formatting
    model_name = "360x"

    model_name += "_" + exp_label

    if args.use_front:
        model_name += '_front'
    if args.use_360:
        model_name += '_360'

    if args.use_clip:
        model_name += '_clip'
    if args.use_audio:
        model_name += '_audio'
    if args.use_audio_tracking:
        model_name += '_audio_tracking'

    model_name += '_useFPS_{}_fusion_{}_optimizer_{}'.format(args.use_video_frames,
                                args.fusion_method,
                                args.optimizer)

    model_name += "_bs_{}_wd_{}".format(args.batch_size, args.weight_decay)

    run = wandb.init(
        # Set the project where this run will be logged
        project=project,
        # group="",
        tags="DEBUG",
        name=model_name,
        # Track hyperparameters and run metadata
        config={
            "dataset": args.dataset,
            "use_front": args.use_front,
            "use_clip": args.use_clip,
            "use_audio": args.use_audio,
            "use_360": args.use_360,
            "use_audio_tracking": args.use_audio_tracking,

            "use_video_frames": args.use_video_frames,
            "optimizer": args.optimizer,
            "fusion_method": args.fusion_method,

            "lr": args.learning_rate,
            "bs": args.batch_size}
    )


    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)
    model.apply(weight_init)  
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if  args.dataset == '360x':
        csvfile = "../prepare/prepare_data/360x/csv/train_files.csv"

        train_dataset = x360Dataset(args, csvfile=csvfile, mode='train', 
                                    use_video_frames=args.use_video_frames,
                                    use_front=args.use_front,
                                    use_360=args.use_360, use_clip=args.use_clip)
        csvfile = "../prepare/prepare_data/360x/csv/test_files.csv"

        if args.test_folder:
            D = x360DatasetTest
            V = valid_folder
        else:
            D = x360Dataset
            V = valid

        test_dataset = D(args, csvfile=csvfile, mode='test',
                               use_video_frames=args.use_video_frames,
                               use_front=args.use_front,
                                use_360=args.use_360, use_clip=args.use_clip)

        # Test

    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support 360x Dataset for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)   #non_blocking=True 

    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=False, num_workers=8, pin_memory=True)  # 

    if args.train:

        best_acc = 0.0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc = V(args, model, device, test_dataloader)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer_dict = {'Total Accuracy': acc['total']}

                writer_dict['Ensemble Accuracy']= acc['ensemble']

                if args.use_front:
                    writer_dict['Front Visual Accuracy'] = acc['v_front']
                    if args.use_audio:
                        writer_dict['Front Audio Accuracy'] = acc['a_front']
                if args.use_360:
                    writer_dict['360 Visual Accuracy'] = acc['v_360']
                    if args.use_audio:
                        writer_dict['360 Audio Accuracy'] = acc['a_360']
                if args.use_clip:
                    writer_dict['Clip Visual Accuracy'] = acc['v_clip']
                    if args.use_audio:
                        writer_dict['Clip Audio Accuracy'] = acc['a_clip']


                writer.add_scalars('Evaluation', writer_dict, epoch)



            else:
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc = V(args, model, device, test_dataloader)


            wandb.log({'Train Loss': batch_loss,
                       'Val Accuracy': acc})

            writer_dict = {'Total Accuracy': acc['total'], }
            writer_dict['Ensemble Accuracy'] = acc['ensemble']
            if args.use_front:
                writer_dict['Front Visual Accuracy'] = acc['v_front']
                if args.use_audio:
                    writer_dict['Front Audio Accuracy'] = acc['a_front']
            if args.use_360:
                writer_dict['360 Visual Accuracy'] = acc['v_360']
                if args.use_audio:
                    writer_dict['360 Audio Accuracy'] = acc['a_360']
            if args.use_clip:
                writer_dict['Clip Visual Accuracy'] = acc['v_clip']
                if args.use_audio:
                    writer_dict['Clip Audio Accuracy'] = acc['a_clip']


            if acc['total'] > best_acc:
                best_acc = float(acc['total'])

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                name = model_name + '_epoch_{}_acc_{}.pth'.format(epoch, acc['total'])

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc['total'],
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, name)

                torch.save(saved_dict, save_dir)
                
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc['total']))
                print(writer_dict)
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc['total'], best_acc))
                print(writer_dict)

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model = model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc = V(args, model, device, test_dataloader)

        writer_dict = {'Total Accuracy': acc['total'], }
        writer_dict['Ensemble Accuracy'] = acc['ensemble']
        if args.use_front:
            writer_dict['Front Visual Accuracy'] = acc['v_front']
            if args.use_audio:
                writer_dict['Front Audio Accuracy'] = acc['a_front']

        if args.use_360:
            writer_dict['360 Visual Accuracy'] = acc['v_360']
            if args.use_audio:
                writer_dict['360 Audio Accuracy'] = acc['a_360']

        if args.use_clip:
            writer_dict['Clip Visual Accuracy'] = acc['v_clip']
            if args.use_audio:
                writer_dict['Clip Audio Accuracy'] = acc['a_clip']


        print(writer_dict)


if __name__ == "__main__":
    main()
