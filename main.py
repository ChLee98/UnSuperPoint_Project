import os
import torch

from torch.utils.data import DataLoader
import torch.optim as optim

import yaml
import argparse
from tqdm import tqdm
import logging

from settings import *

from utils.loader import dataLoader, testLoader
from model import UnSuperPoint

###### util functions ######
def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*config['training']['batch_size_train'], len(train_loader)))
    pass

def simple_train(config, output_dir, args):
    batch_size = config['training']['batch_size_train']
    epochs = config['training']['epoch_train']
    learning_rate = config['training']['learning_rate']
    savepath = os.path.join(output_dir, 'checkpoints')
    os.makedirs(savepath, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Prepare for data loader
    # dataset = Picture(config, transform)
    # trainloader = DataLoader(dataset, batch_size=batch_size,
    #                     num_workers=config['training']['workers_train'],
    #                     shuffle=True, drop_last=True)
    data = dataLoader(config, dataset=config['data']['dataset'], warp_input=True)
    trainloader, valloader = data['train_loader'], data['val_loader']

    datasize(trainloader, config, tag='train')
    datasize(valloader, config, tag='val')

    # Prepare for model
    model = UnSuperPoint(config)
    model.train()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model.to(dev)

    # Do optimization
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    whole_step = 0
    total = len(trainloader)
    try:
        for epoch in tqdm(range(1, epochs+1), desc='epoch'):
            error = 0
            for batch_idx, (img0, img1, mat) in tqdm(enumerate(trainloader), desc='step', total=total):
                whole_step += 1

                # print(img0.shape,img1.shape)
                img0 = img0.to(dev)
                img1 = img1.to(dev)
                mat = mat.squeeze()
                mat = mat.to(dev)                     
                optimizer.zero_grad()
                s1,p1,d1 = model(img0)
                s2,p2,d2 = model(img1)
                # TODO: All code does not consider batch_size larger than 1
                s1 = torch.squeeze(s1, 0); s2 = torch.squeeze(s2, 0)
                p1 = torch.squeeze(p1, 0); p2 = torch.squeeze(p2, 0)
                d1 = torch.squeeze(d1, 0); d2 = torch.squeeze(d2, 0)
                # print(s1.shape,s2.shape,p1.shape,p2.shape,d1.shape,d2.shape,mat.shape)
                # loss = model.UnSuperPointLoss(s1,p1,d1,s2,p2,d2,mat)
                loss = model.loss(s1,p1,d1,s2,p2,d2,mat)
                loss.backward()
                optimizer.step()
                error += loss.item()

                tqdm.write('Loss: {:.6f}'.format(error))
                
                if whole_step % config['save_interval'] == 0:
                    torch.save(model.state_dict(), os.path.join(savepath, config['model']['name'] + '_{}.pkl'.format(whole_step)))
                
                if args.eval and whole_step % config['validation_interval'] == 0:
                    # TODO: Validation code should be implemented
                    pass

                error = 0

        torch.save(model.state_dict(), os.path.join(savepath, config['model']['name'] + '_{}.pkl'.format(whole_step)))

    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")
        torch.save(model.state_dict(), os.path.join(savepath, config['model']['name'] + '_{}.pkl'.format(whole_step)))
        pass

@torch.no_grad()
def simple_test(config, output_dir, args):
    model_path = os.path.join(output_dir, 'checkpoints', args.model_name)
    savepath = os.path.join(output_dir, 'results')
    os.makedirs(savepath, exist_ok=True)

    # Prepare for data loader
    data = testLoader(config, dataset=config['data']['dataset'], warp_input=True)
    testloader = data['test_loader']
    
    # Prepare for model
    model = UnSuperPoint()
    model.load_state_dict(torch.load(model_path))
    model.to(model.dev)
    model.train(False)

    # Do prediction
    model.predict(os.path.join(config['data']['root'], HPatches_SRC),\
                os.path.join(config['data']['root'], HPatches_DST), savepath)

if __name__ == '__main__':
    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('export_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=simple_train)

    # Testing command
    p_test = subparsers.add_parser('test')
    p_test.add_argument('config', type=str)
    p_test.add_argument('export_name', type=str)
    p_test.add_argument('model_name', type=str)
    p_test.set_defaults(func=simple_test)
    
    args = parser.parse_args()

    output_dir = os.path.join(EXPORT_PATH, args.export_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    args.func(config, output_dir, args)
    
