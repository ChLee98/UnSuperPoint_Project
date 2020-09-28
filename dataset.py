import cv2
import os
import random

from PIL import Image
import numpy as np

import torchvision.transforms as T
from torch.utils.data import Dataset

from settings import COCO_TRAIN, COCO_VAL

transform = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.225,0.225,0.225])
        ])

transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.225,0.225,0.225])
        ])

class Picture(Dataset):
    def __init__(self, config, transforms=None, train=True):
        self.train = train
        self.config = config
        root = os.path.join(config['data']['root'], COCO_TRAIN)
        images = os.listdir(root)
        self.images = [os.path.join(root, image) for image in images if image.endswith('.jpg')]
        self.transforms = transforms  

    def __getitem__(self,index):
        image_path = self.images[index]
        # print(image_path)
        cv_image = cv2.imread(image_path)
        # print(cv_image.shape,image_path)
        # cv_image = Image.fromarray(cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB))
        re_img = self.Myresize(cv_image)
        # re_img = cv2.resize(cv_image,(config['data']['resize'][1],
        #     config['data']['resize'][0]))
        # re_img = transform_handle(cv_image)
        # re_img = cv2.cvtColor(np.asarray(re_img),cv2.COLOR_RGB2BGR)
        tran_img, tran_mat = self.EnhanceData(re_img)

        # cv2.imwrite('re_+' + str(index) +'.jpg',re_img)
        # cv2.imwrite('tran_'+ str(index) +'.jpg',tran_img)
        if self.transforms:
            re_img = Image.fromarray(cv2.cvtColor(re_img,cv2.COLOR_BGR2RGB))
            source_img = transform_test(re_img)
            tran_img = Image.fromarray(cv2.cvtColor(tran_img,cv2.COLOR_BGR2RGB))
            des_img = self.transforms(tran_img)
        # else:
        #     re_img = Image.fromarray(cv2.cvtColor(re_img,cv2.COLOR_BGR2RGB))
        #     tran_img = Image.fromarray(cv2.cvtColor(tran_img,cv2.COLOR_BGR2RGB))
        #     image_array1 = np.asarray(re_img)
        #     image_array2 = np.asarray(tran_img)
        #     source_img = torch.from_numpy(image_array1)
        #     des_img = torch.from_numpy(image_array2)
        # if self.train:
        return source_img, des_img, tran_mat
        # else:
            # return source_img,des_img,tran_mat
            
    def __len__(self):
        return len(self.images)

    def Myresize(self, img):
        # print(img.shape)
        h,w = img.shape[:2]
        if h < self.config['data']['resize'][0] or w < self.config['data']['resize'][1]:
            new_h = self.config['data']['resize'][0]
            new_w = self.config['data']['resize'][1]
            h = new_h
            w = new_w
            img = cv2.resize(img,(new_w, new_h))
            
        new_h, new_w = self.config['data']['resize']
        try:
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
        except:
            print(h,new_h,w,new_w)
            raise 
        img = img[top: top + new_h,
                            left: left + new_w]
        return img

    def EnhanceData(self, img):
        seed = random.randint(1,20)
        src_point =np.array( [(0,0),
            (self.config['data']['resize'][1]-1, 0),
            (0, self.config['data']['resize'][0]-1),
            (self.config['data']['resize'][1]-1, self.config['data']['resize'][0]-1)],
            dtype = 'float32')

        dst_point = self.get_dst_point()
        center = (self.config['data']['resize'][1]/2, self.config['data']['resize'][0]/2)
        rot = random.randint(-2,2) * self.config['data']['rot'] + random.randint(0,15)
        scale = 1.2 - self.config['data']['scale']*random.random()
        RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
        f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
        mat = cv2.getPerspectiveTransform(src_point, f_point)
        out_img = cv2.warpPerspective(img, mat,(self.config['data']['resize'][1],self.config['data']['resize'][0]))
        if seed > 10 and seed <= 15:
            out_img = cv2.GaussianBlur(out_img, (3, 3), sigmaX=0)
        return out_img,mat

    def get_dst_point(self):
        a = random.random()
        b = random.random()
        c = random.random()
        d = random.random()
        e = random.random()
        f = random.random()

        if random.random() > 0.5:
            left_top_x = self.config['data']['perspective']*a
            left_top_y = self.config['data']['perspective']*b
            right_top_x = 0.9 + self.config['data']['perspective']*c
            right_top_y = self.config['data']['perspective']*d
            left_bottom_x  = self.config['data']['perspective']*a
            left_bottom_y  = 0.9 + self.config['data']['perspective']*e
            right_bottom_x = 0.9 + self.config['data']['perspective']*c
            right_bottom_y = 0.9 + self.config['data']['perspective']*f
        else:
            left_top_x = self.config['data']['perspective']*a
            left_top_y = self.config['data']['perspective']*b
            right_top_x = 0.9+self.config['data']['perspective']*c
            right_top_y = self.config['data']['perspective']*d
            left_bottom_x  = self.config['data']['perspective']*e
            left_bottom_y  = 0.9 + self.config['data']['perspective']*b
            right_bottom_x = 0.9 + self.config['data']['perspective']*f
            right_bottom_y = 0.9 + self.config['data']['perspective']*d

        # left_top_x = config['data']['perspective']*random.random()
        # left_top_y = config['data']['perspective']*random.random()
        # right_top_x = 0.9+config['data']['perspective']*random.random()
        # right_top_y = config['data']['perspective']*random.random()
        # left_bottom_x  = config['data']['perspective']*random.random()
        # left_bottom_y  = 0.9 + config['data']['perspective']*random.random()
        # right_bottom_x = 0.9 + config['data']['perspective']*random.random()
        # right_bottom_y = 0.9 + config['data']['perspective']*random.random()

        dst_point = np.array([(self.config['data']['resize'][1]*left_top_x,self.config['data']['resize'][0]*left_top_y,1),
                (self.config['data']['resize'][1]*right_top_x, self.config['data']['resize'][0]*right_top_y,1),
                (self.config['data']['resize'][1]*left_bottom_x,self.config['data']['resize'][0]*left_bottom_y,1),
                (self.config['data']['resize'][1]*right_bottom_x,self.config['data']['resize'][0]*right_bottom_y,1)],dtype = 'float32')
        return dst_point
