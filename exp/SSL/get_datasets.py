
from torchvision import transforms
from datasets.x360dataset import x360Dataset
from torch.utils.data import DataLoader, random_split


def get_dataset(args):
    if  args.dataset == '360x':
        csvfile = "../prepare/prepare_data/360x/csv/train_files.csv"
        train_dataset = x360Dataset(args, csvfile=csvfile, mode='train',
                                    use_front=args.use_front,
                                    use_360=args.use_360, use_clip=args.use_clip)

        train_dataloader = DataLoader(train_dataset, batch_size=args.bs,
                                      shuffle=True, num_workers=args.workers, pin_memory=True)  # non_blocking=True
        csvfile = "../prepare/prepare_data/360x/csv/val_files.csv"

        val_dataset = x360Dataset(args, csvfile=csvfile, mode='test',
                         use_front=args.use_front,
                         use_360=args.use_360, use_clip=args.use_clip)



        val_dataloader = DataLoader(val_dataset, batch_size=args.bs,
                                 shuffle=False, num_workers=args.workers, pin_memory=True)  #

        csvfile = "../prepare/prepare_data/360x/csv/test_files.csv"

        test_dataset = x360Dataset(args, csvfile=csvfile, mode='test',
                         use_front=args.use_front,
                         use_360=args.use_360, use_clip=args.use_clip)



        test_dataloader = DataLoader(test_dataset, batch_size=args.bs,
                                 shuffle=False, num_workers=args.workers, pin_memory=True)  #



    return train_dataloader, val_dataloader, test_dataloader