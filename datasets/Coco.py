import cv2
import os
import random

from PIL import Image
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from utils.tools import dict_update, Myresize
from utils.utils import sample_homography_np as sample_homography, inv_warp_image

from settings import COCO_TRAIN, COCO_VAL, DATA_PATH

class Coco(Dataset):
    default_config = {


    }
    def __init__(self, transform=None, task='train', **config):
        self.task = task
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        root = Path(DATA_PATH, 'COCO/' + task + '2014/')
        images = list(root.iterdir())
        self.images = [str(p) for p in images]
        # root = os.path.join(config['root'], COCO_TRAIN)
        # images = os.listdir(root)
        # self.images = [os.path.join(root, image) for image in images if image.endswith('.jpg')]
        self.transforms = transform

    def __getitem__(self, index):
        if self.task == 'train':
            image_path = self.images[index]
            cv_image = cv2.imread(image_path)
            re_img = Myresize(cv_image, self.config['resize'])
            tran_img = self.EnhanceData(re_img)

            if self.transforms:
                re_img = Image.fromarray(cv2.cvtColor(re_img,cv2.COLOR_BGR2RGB))
                source_img = self.transforms(re_img)

                tran_img = Image.fromarray(cv2.cvtColor(tran_img,cv2.COLOR_BGR2RGB))
                des_img = self.transforms(tran_img)
        else: # self.task == 'valid'
            # TODO: Should be implemented
            pass

        tran_mat = sample_homography(np.array([2, 2]), shift=-1, **self.config['homographies'])
        mat = torch.tensor(tran_mat, dtype=torch.float32)
        inv_mat = torch.inverse(mat)
        des_img = inv_warp_image(des_img, inv_mat).squeeze(0)

        # H, W = self.config['resize']
        # norm = torch.tensor([[2/W, 0, -1], [0, 2/H, -1], [0, 0, 1]], dtype=torch.float32)
        # denorm = torch.tensor([[W, 0, W], [0, H, H], [0, 0, 2]], dtype=torch.float32)
        # mat = denorm * mat * norm

        # return source_img, des_img, tran_mat
        return source_img, des_img, mat
            
    def __len__(self):
        return len(self.images)

    def EnhanceData(self, img):
        seed = random.randint(1,20)
        if seed > 10 and seed <= 15:
            img = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
        return img
