import cv2
import os
import random
import math

from PIL import Image
import numpy as np
from pathlib import Path

import torch
from datasets.Coco import Coco
from utils.tools import dict_update, Myresize
from utils.utils import sample_homography_np as sample_homography, inv_warp_image

from settings import COCO_TRAIN, COCO_VAL, DATA_PATH

class CocoNutV2(Coco):
    def __init__(self, transform=None, task='train', **config):
        super(CocoNutV2, self).__init__(transform, task, **config)

        root = Path(DATA_PATH, 'COCO/' + task + '2014/')
        images = list(root.iterdir())
        self.images = [[str(p), str(q)] for p, q in zip(images, random.sample(images, len(images)))]

    def __getitem__(self, index):
        img1_path, img2_path = self.images[index]
        cv_img1 = cv2.imread(img1_path)
        cv_img2 = cv2.imread(img2_path)
        re_img1 = Myresize(cv_img1, self.config['resize'])
        re_img2 = Myresize(cv_img2, self.config['resize'])
        tran_img1 = self.EnhanceData(re_img1)
        tran_img2 = self.EnhanceData(re_img2)

        mask = self.generate_mask(self.config['resize'])
        bmask = torch.ones(self.config['resize'], dtype=torch.float32)

        if self.transforms:
            # re_img1 = Image.fromarray(cv2.cvtColor(re_img1, cv2.COLOR_BGR2RGB))
            # source_img = self.transforms(re_img1)

            tran_img1 = Image.fromarray(cv2.cvtColor(tran_img1,cv2.COLOR_BGR2RGB))
            tran_img1 = self.transforms(tran_img1)

            tran_img2 = Image.fromarray(cv2.cvtColor(tran_img2,cv2.COLOR_BGR2RGB))
            tran_img2 = self.transforms(tran_img2)

        mat1 = sample_homography(np.array([2, 2]), shift=-1, **self.config['homographies'])
        mat2 = sample_homography(np.array([2, 2]), shift=-1, **self.config['homographies'])
        mat1 = torch.tensor(mat1, dtype=torch.float32)
        mat2 = torch.tensor(mat2, dtype=torch.float32)

        # inv_mat1 = torch.inverse(mat1)
        # tran_img1 = inv_warp_image(tran_img1, inv_mat1).squeeze(0)

        inv_mat2 = torch.inverse(mat2)
        tran_img2 = inv_warp_image(tran_img2, inv_mat2).squeeze(0)

        mask = inv_warp_image(torch.stack([mask, mask, mask]), inv_mat2)[0].unsqueeze(0)
        cmask = bmask.unsqueeze(0) - mask
        # bmask = inv_warp_image(torch.stack([bmask, bmask, bmask]), inv_mat1)[0].unsqueeze(0)

        source_img = cmask*tran_img1 + mask*tran_img2

        inv_mat1 = torch.inverse(mat1)
        des_img = inv_warp_image(source_img, inv_mat1).squeeze(0)

        return source_img, des_img, mat1

    def generate_mask(self, size = [240, 320]):
        # Prepare a canvas
        mask = np.zeros([size[0], size[1], 3], np.uint8)

        n = random.randint(3, 8)
        lenmean = min(size)
        for i in range(n):
            direction = -math.pi/2
            pts = [(random.randrange(size[1]), random.randrange(size[0]))]
            m = random.randint(2, 6)
            for j in range(m):
                length = random.uniform(lenmean*0.075, lenmean*0.125)
                direction += random.uniform(0, math.pi/2)
                vector = (int(length * math.cos(direction)) + pts[j][0],
                            int(length * math.sin(direction)) + pts[j][1])

                pts.append(vector)

            # Draw a polygon
            cv2.fillPoly(mask, [np.array(pts, np.int32)], (255, 255, 255))

        return torch.tensor(mask[:,:,0]/255, dtype=torch.float32)