import cv2
import os
import random

from PIL import Image
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from utils.tools import dict_update, Myresize

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
            tran_img, tran_mat = self.EnhanceData(re_img)
            if self.transforms:
                re_img = Image.fromarray(cv2.cvtColor(re_img,cv2.COLOR_BGR2RGB))
                source_img = self.transforms(re_img)

                tran_img = Image.fromarray(cv2.cvtColor(tran_img,cv2.COLOR_BGR2RGB))
                des_img = self.transforms(tran_img)
        else: # self.task == 'valid'
            # TODO: Should be implemented
            pass

        return source_img, des_img, tran_mat
            
    def __len__(self):
        return len(self.images)

    def EnhanceData(self, img):
        seed = random.randint(1,20)
        src_point =np.array( [(0,0),
            (self.config['resize'][1]-1, 0),
            (0, self.config['resize'][0]-1),
            (self.config['resize'][1]-1, self.config['resize'][0]-1)],
            dtype = 'float32')

        dst_point = self.get_dst_point()
        center = (self.config['resize'][1]/2, self.config['resize'][0]/2)
        rot = random.randint(-2,2) * self.config['rot'] + random.randint(0,15)
        scale = 1.2 - self.config['scale']*random.random()
        RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
        f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
        mat = cv2.getPerspectiveTransform(src_point, f_point)
        out_img = cv2.warpPerspective(img, mat,(self.config['resize'][1],self.config['resize'][0]))
        if seed > 10 and seed <= 15:
            out_img = cv2.GaussianBlur(out_img, (3, 3), sigmaX=0)
        return out_img, mat

    def get_dst_point(self):
        a = random.random()
        b = random.random()
        c = random.random()
        d = random.random()
        e = random.random()
        f = random.random()

        if random.random() > 0.5:
            left_top_x = self.config['perspective']*a
            left_top_y = self.config['perspective']*b
            right_top_x = 0.9 + self.config['perspective']*c
            right_top_y = self.config['perspective']*d
            left_bottom_x  = self.config['perspective']*a
            left_bottom_y  = 0.9 + self.config['perspective']*e
            right_bottom_x = 0.9 + self.config['perspective']*c
            right_bottom_y = 0.9 + self.config['perspective']*f
        else:
            left_top_x = self.config['perspective']*a
            left_top_y = self.config['perspective']*b
            right_top_x = 0.9+self.config['perspective']*c
            right_top_y = self.config['perspective']*d
            left_bottom_x  = self.config['perspective']*e
            left_bottom_y  = 0.9 + self.config['perspective']*b
            right_bottom_x = 0.9 + self.config['perspective']*f
            right_bottom_y = 0.9 + self.config['perspective']*d

        dst_point = np.array([(self.config['resize'][1]*left_top_x,self.config['resize'][0]*left_top_y,1),
                (self.config['resize'][1]*right_top_x, self.config['resize'][0]*right_top_y,1),
                (self.config['resize'][1]*left_bottom_x,self.config['resize'][0]*left_bottom_y,1),
                (self.config['resize'][1]*right_bottom_x,self.config['resize'][0]*right_bottom_y,1)],dtype = 'float32')
        return dst_point
