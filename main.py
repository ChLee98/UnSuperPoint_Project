import os
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

import yaml
import argparse
from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path

from settings import *

from model_wrap import PointTracker

from utils.loader import dataLoader, testLoader
from utils.utils import getWriterPath
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
    data = dataLoader(config, dataset=config['data']['dataset'], warp_input=True)
    trainloader, valloader = data['train_loader'], data['val_loader']

    datasize(trainloader, config, tag='train')
    datasize(valloader, config, tag='val')

    # Prepare for model
    model = UnSuperPoint(config)
    model.train()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model.to(dev)

    # Prepare for tensorboard writer
    model.writer = SummaryWriter(getWriterPath(task=args.command, 
        exper_name=args.export_name, date=True))

    # Prepare for optimizer
    # model.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    whole_step = 0
    total = len(trainloader)
    try:
        for epoch in tqdm(range(1, epochs+1), desc='epoch'):
            for batch_idx, (img0, img1, mat) in tqdm(enumerate(trainloader), desc='step', total=total):
                whole_step += 1
                model.step = whole_step

                loss = model.train_val_step(img0, img1, mat, 'train')

                tqdm.write('Loss: {:.6f}'.format(loss))
                
                if whole_step % config['save_interval'] == 0:
                    torch.save(model.state_dict(), os.path.join(savepath, config['model']['name'] + '_{}.pkl'.format(whole_step)))
                
                if args.eval and whole_step % config['validation_interval'] == 0:
                    # TODO: Validation code should be implemented
                    pass

        torch.save(model.state_dict(), os.path.join(savepath, config['model']['name'] + '_{}.pkl'.format(whole_step)))

    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")
        torch.save(model.state_dict(), os.path.join(savepath, config['model']['name'] + '_{}.pkl'.format(whole_step)))
        pass

def simple_export(config, output_dir, args):
    """
    # input 2 images, output keypoints and correspondence
    save prediction:
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N1, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)
            'matches': np [N3, 4]
    """
    from utils.loader import get_save_path
    from utils.tools import squeezeToNumpy

    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    save_path = get_save_path(output_dir)
    save_output = save_path / "../predictions"
    os.makedirs(save_output, exist_ok=True)
    model_path = os.path.join(output_dir, 'checkpoints', args.model_name)

    ## parameters
    outputMatches = True

    # data loading
    from utils.loader import testLoader as dataLoader
    task = config["data"]["dataset"]
    data = dataLoader(config, dataset=task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    # model loading
    # from utils.loader import get_module
    # Val_model = get_module("", config["front_end_model"])

    ## load pretrained
    # val_agent = Val_model()
    val_agent = UnSuperPoint()
    val_agent.load_state_dict(torch.load(model_path))
    val_agent.to(val_agent.dev)
    val_agent.train(False)
    val_agent.task = 'test'

    ## tracker
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    ###### check!!!
    count = 0
    total = len(test_loader)
    for i, sample in tqdm(enumerate(test_loader), desc='scene', total=total):
        img_0, img_1 = sample["image"], sample["warped_image"]

        # first image, no matches
        # img = img_0
        @torch.no_grad()
        def get_pts_desc_from_agent(val_agent, img, device="cpu"):
            """
            pts: list [numpy (3, N)]
            desc: list [numpy (256, N)]
            """
            score, point, desc = val_agent.forward(img.to(device)) # heatmap: numpy [batch, 1, H, W]
            point = val_agent.get_position(point[0])
            # heatmap, pts to pts, desc
            pts, desc = val_agent.getPtsDescFromHeatmap(point.cpu().numpy(), score[0].cpu().numpy(), desc[0].cpu().numpy())

            outs = {"pts": pts, "desc": desc}
            return outs

        def transpose_np_dict(outs):
            for entry in list(outs):
                outs[entry] = outs[entry].transpose()

        outs = get_pts_desc_from_agent(val_agent, img_0, device=device)
        pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]

        tqdm.write("pts A: {}, desc A: {}".format(pts.shape, desc.shape))

        if outputMatches == True:
            tracker.update(pts, desc)

        # save keypoints
        img_0 = squeezeToNumpy(img_0).transpose((1, 2, 0))
        pred = {"image": ((img_0/0.225+0.5)*255).astype(np.uint8)}
        pred.update({"prob": pts.transpose(), "desc": desc.transpose()})

        # second image, output matches
        outs = get_pts_desc_from_agent(val_agent, img_1, device=device)
        pts, desc = outs["pts"], outs["desc"]

        tqdm.write("pts B: {}, desc B: {}".format(pts.shape, desc.shape))

        if outputMatches == True:
            tracker.update(pts, desc)

        img_1 = squeezeToNumpy(img_1).transpose((1, 2, 0))
        pred.update({"warped_image": ((img_1/0.225+0.5)*255).astype(np.uint8)})
        # print("total points: ", pts.shape)
        pred.update(
            {
                "warped_prob": pts.transpose(),
                "warped_desc": desc.transpose(),
                "homography": squeezeToNumpy(sample["homography"]),
            }
        )

        if outputMatches == True:
            matches = tracker.get_matches()
            tqdm.write("matches: {}".format(matches.transpose().shape))
            pred.update({"matches": matches.transpose()})

        # clean last descriptor
        tracker.clear_desc()

        filename = str(count)
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)
        # print("save: ", path)
        count += 1
    print("output pairs: ", count)

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
    # TODO: Make predictions done for all images in HPatches dataset.
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

    # Export command
    p_export = subparsers.add_parser('export')
    p_export.add_argument("config", type=str)
    p_export.add_argument("export_name", type=str)
    p_export.add_argument('model_name', type=str)
    p_export.add_argument("--correspondence", action="store_true")
    p_export.add_argument("--eval", action="store_true")
    p_export.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_export.set_defaults(func=simple_export)

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
    
