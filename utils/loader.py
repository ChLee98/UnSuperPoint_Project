"""many loaders
# loader for model, dataset, testing dataset
"""

import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim
import torch.utils.data

# from settings import EXPER_PATH

# from utils.loader import get_save_path
def get_save_path(output_dir):
    """
    This func
    :param output_dir:
    :return:
    """
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.

   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

   """
   base_seed = torch.IntTensor(1).random_().item()
   # print(worker_id, base_seed)
   np.random.seed(base_seed + worker_id)


def dataLoader(config, dataset='Coco', warp_input=False, train=True, val=True):
    import torchvision.transforms as transforms
    training_params = config.get('training', {})
    workers_train = training_params.get('workers_train', 1) # 16
    workers_val   = training_params.get('workers_val', 1) # 16
        
    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}")
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[1/0.225,1/0.225,1/0.225])
        ]),
        'val': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[1/0.225,1/0.225,1/0.225])
        ]),
    }
    Dataset = get_module('datasets', dataset)
    print(f"dataset: {dataset}")

    train_set = Dataset(
        transform=data_transforms['train'],
        task = 'train',
        **config['data'],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['training']['batch_size_train'], shuffle=True,
        pin_memory=True,
        num_workers=workers_train,
        worker_init_fn=worker_init_fn
    )
    val_set = Dataset(
        transform=data_transforms['val'],
        task = 'val',
        **config['data'],
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config['training']['batch_size_val'], shuffle=True,
        pin_memory=True,
        num_workers=workers_val,
        worker_init_fn=worker_init_fn
    )
    # val_set, val_loader = None, None
    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}

def testLoader(config, dataset='hpatches', warp_input=False):
    import torchvision.transforms as transforms
    training_params = config.get('testing', {})
    workers_test = training_params.get('workers_test', 1) # 16
    logging.info(f"workers_test: {workers_test}")
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[1/0.225,1/0.225,1/0.225])
        ])
    }
    test_loader = None
    if dataset == 'hpatches':
        from datasets.patches_dataset import PatchesDataset
        if config['data']['preprocessing']['resize']:
            size = config['data']['preprocessing']['resize']
        test_set = PatchesDataset(
            transform=data_transforms['test'],
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn
        )
    return {'test_set': test_set, 'test_loader': test_loader}

def get_module(path, name):
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)

