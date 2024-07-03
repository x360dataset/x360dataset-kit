import torch, os, random
import numpy as np

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(cfg, name, mode, **kwargs):
   """
       A simple dataset builder
   """
   print("datasets[name]:", name)
   dataset = datasets[name](cfg, mode, **kwargs)
   return dataset

def make_data_loader(dataset, mode, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if mode=="train" else None),
        shuffle=mode=="train",
        drop_last=mode=="train",
        generator=generator,
        persistent_workers=True
    )
    return loader

def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)