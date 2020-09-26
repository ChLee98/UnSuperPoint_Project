import os
import random

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T 
import torch.optim as optim

import yaml
import argparse

from settings import EXPORT_PATH, COCO_TRAIN, COCO_VAL

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
        rot = random.randint(-2,2) * config['data']['rot'] + random.randint(0,15)
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
            left_top_x = config['data']['perspective']*a
            left_top_y = config['data']['perspective']*b
            right_top_x = 0.9 + config['data']['perspective']*c
            right_top_y = config['data']['perspective']*d
            left_bottom_x  = config['data']['perspective']*a
            left_bottom_y  = 0.9 + config['data']['perspective']*e
            right_bottom_x = 0.9 + config['data']['perspective']*c
            right_bottom_y = 0.9 + config['data']['perspective']*f
        else:
            left_top_x = config['data']['perspective']*a
            left_top_y = config['data']['perspective']*b
            right_top_x = 0.9+config['data']['perspective']*c
            right_top_y = config['data']['perspective']*d
            left_bottom_x  = config['data']['perspective']*e
            left_bottom_y  = 0.9 + config['data']['perspective']*b
            right_bottom_x = 0.9 + config['data']['perspective']*f
            right_bottom_y = 0.9 + config['data']['perspective']*d

        # left_top_x = config['data']['perspective']*random.random()
        # left_top_y = config['data']['perspective']*random.random()
        # right_top_x = 0.9+config['data']['perspective']*random.random()
        # right_top_y = config['data']['perspective']*random.random()
        # left_bottom_x  = config['data']['perspective']*random.random()
        # left_bottom_y  = 0.9 + config['data']['perspective']*random.random()
        # right_bottom_x = 0.9 + config['data']['perspective']*random.random()
        # right_bottom_y = 0.9 + config['data']['perspective']*random.random()

        dst_point = np.array([(config['data']['resize'][1]*left_top_x,config['data']['resize'][0]*left_top_y,1),
                (config['data']['resize'][1]*right_top_x, config['data']['resize'][0]*right_top_y,1),
                (config['data']['resize'][1]*left_bottom_x,config['data']['resize'][0]*left_bottom_y,1),
                (config['data']['resize'][1]*right_bottom_x,config['data']['resize'][0]*right_bottom_y,1)],dtype = 'float32')
        return dst_point

class UnSuperPoint(nn.Module):
    def __init__(self, config):
        super(UnSuperPoint, self).__init__()
        self.usp = config['model']['usp_loss']['alpha_usp']
        self.position_weight = config['model']['usp_loss']['alpha_position']
        self.score_weight = config['model']['usp_loss']['alpha_score']
        self.uni_xy = config['model']['unixy_loss']['alpha_unixy']
        self.desc = config['model']['desc_loss']['alpha_desc']
        self.d = config['model']['desc_loss']['lambda_d']
        self.m_p = config['model']['desc_loss']['margin_positive']
        self.m_n = config['model']['desc_loss']['margin_negative']
        self.decorr = config['model']['decorr_loss']['alpha_decorr']
        self.correspond = config['model']['correspondence_threshold']

        self.downsample = 8
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.score = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256,1,3,1,padding=1),
            nn.Sigmoid()
        )
        self.position = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256,2,3,1,padding=1),
            nn.Sigmoid()
        )
        self.descriptor = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256, 256,3,1,padding=1)
        )

    def forward(self, x):
        h,w = x.shape[-2:]
        self.h = h
        self.w = w
        output = self.cnn(x)
        s = self.score(output)
        p = self.position(output)
        d = self.descriptor(output)
        desc = self.interpolate(p, d, h, w)
        return s,p,desc

    def interpolate(self, p, d, h, w):
        # b, c, h, w
        # h, w = p.shape[2:]
        samp_pts = self.get_bath_position(p)
        samp_pts[:, 0, :, :] = (samp_pts[:, 0, :, :] / (float(self.w)/2.)) - 1.
        samp_pts[:, 1, :, :] = (samp_pts[:, 1, :, :] / (float(self.h)/2.)) - 1.
        samp_pts = samp_pts.permute(0,2,3,1)
        desc = torch.nn.functional.grid_sample(d, samp_pts)
        return desc

    def loss(self, bath_As, bath_Ap, bath_Ad, 
        bath_Bs, bath_Bp, bath_Bd, mat):
        loss = 0
        bath = bath_As.shape[0]
        for i in range(bath):
            loss += self.UnSuperPointLoss(bath_As[i], bath_Ap[i], bath_Ad[i], 
        bath_Bs[i], bath_Bp[i], bath_Bd[i],mat[i])
        return loss / bath

    def UnSuperPointLoss(self, As, Ap, Ad, Bs, Bp, Bd, mat):
        position_A = self.get_position(Ap, flag='A', mat=mat)
        position_B = self.get_position(Bp, flag='B', mat=None)
        # position_A = self.get_batch_position(Ap, flag='A', mat=mat)
        # position_B = self.get_batch_position(Bp, flag='B', mat=None)
        G = self.getG(position_A,position_B)

        Usploss = self.usploss(As, Bs, mat, G)
        Uni_xyloss = self.uni_xyloss(Ap, Bp)
        
        Descloss = self.descloss(Ad, Bd, G)
        Decorrloss = self.decorrloss(Ad, Bd)
        return (self.usp * Usploss + self.uni_xy * Uni_xyloss +
            self.desc * Descloss + self.decorr *Decorrloss)

    def usploss(self, As, Bs, mat, G):
        reshape_As_k, reshape_Bs_k, d_k = self.get_point_pair(
            G, As, Bs)
        # print(d_k)
        # print(reshape_As_k.shape,reshape_Bs_k.shape,d_k.shape)
        positionK_loss = torch.mean(d_k)
        scoreK_loss = torch.mean(torch.pow(reshape_As_k - reshape_Bs_k, 2))
        uspK_loss = self.get_uspK_loss(d_k, reshape_As_k, reshape_Bs_k)        
        return (self.position_weight * positionK_loss + 
            self.score_weight * scoreK_loss + uspK_loss)

    def get_bath_position(self, Pamp):
        x = 0
        y = 1
        res = torch.zeros_like(Pamp)
        for i in range(Pamp.shape[3]):
            res[:,x,:,i] = (i + Pamp[:,x,:,i]) * self.downsample
        for i in range(Pamp.shape[2]):
            res[:,y,i,:] = (i + Pamp[:,y,i,:]) * self.downsample
        return res

    def get_batch_position(self, Pamp, flag=None, mat=None):
        x = 0
        y = 1
        res = torch.zeros_like(Pamp)
        for i in range(Pamp.shape[3]):
            res[:,x,:,i] = (i + Pamp[:,x,:,i]) * self.downsample
        for i in range(Pamp.shape[2]):
            res[:,y,i,:] = (i + Pamp[:,y,i,:]) * self.downsample
        if flag == 'A':
            r = torch.cat((Pamp, torch.ones((Pamp.shape[0], 1, Pamp.shape[2], Pamp.shape[3])).to(self.dev)), 1).permute(0, 2, 3, 1)
            r = torch.matmul(r, mat.T.unsqueeze(1).unsqueeze(1).float())
            r = torch.div(r,r[:,:,:,2].unsqueeze(3))
            return r
        return res

    def get_position(self, Pmap, flag=None, mat=None):
        x = 0
        y = 1
        res = torch.zeros_like(Pmap)
        # print(Pmap.shape,res.shape)
        for i in range(Pmap.shape[2]):
            res[x,:,i] = (i + Pmap[x,:,i]) * self.downsample
        for i in range(Pmap.shape[1]):
            res[y,i,:] = (i + Pmap[y,i,:]) * self.downsample 
        if flag == 'A':
            # print(mat.shape)
            r = torch.zeros_like(res)
            Denominator = res[x,:,:]*mat[2,0] + res[y,:,:]*mat[2,1] +mat[2,2]
            r[x,:,:] = (res[x,:,:]*mat[0,0] + 
                res[y,:,:]*mat[0,1] +mat[0,2]) / Denominator 
            r[y,:,:] = (res[x,:,:]*mat[1,0] + 
                res[y,:,:]*mat[1,1] +mat[1,2]) / Denominator
            return r
        else:
            return res

    def getG(self, PA, PB):
        c = PA.shape[0]
        # reshape_PA shape = m,c
        reshape_PA = PA.reshape((c,-1)).permute(1,0)
        # reshape_PB shape = m,c
        reshape_PB = PB.reshape((c,-1)).permute(1,0)
        # x shape m,m <- (m,1 - 1,m) 
        x = torch.unsqueeze(reshape_PA[:,0],1) - torch.unsqueeze(reshape_PB[:,0],0)
        # y shape m,m <- (m,1 - 1,m)
        y = torch.unsqueeze(reshape_PA[:,1],1) - torch.unsqueeze(reshape_PB[:,1],0)

        G = torch.sqrt(torch.pow(x,2) + torch.pow(y,2))

        return G

    def get_point_pair(self, G, As, Bs):
        A2B_min_Id = torch.argmin(G,dim=1)
        M = len(A2B_min_Id)
        Id = G[list(range(M)),A2B_min_Id] <= self.correspond
        reshape_As = As.reshape(-1)
        reshape_Bs = Bs.reshape(-1)
        return (reshape_As[Id], reshape_Bs[A2B_min_Id[Id]], 
            G[Id,A2B_min_Id[Id]])

    def get_uspK_loss(self, d_k, reshape_As_k, reshape_Bs_k):
        sk_ = (reshape_As_k + reshape_Bs_k) / 2
        d_ = torch.mean(d_k)
        return torch.mean(sk_ * (d_k - d_))

    def uni_xyloss(self, Ap, Bp):
        c = Ap.shape[0]
        reshape_PA = Ap.reshape((c,-1)).permute(1,0)
        reshape_PB = Bp.reshape((c,-1)).permute(1,0)
        loss = 0
        for i in range(2):
            loss += self.get_uni_xy(reshape_PA[:,i])
            loss += self.get_uni_xy(reshape_PB[:,i])
        return loss
        
    def get_uni_xy(self, position):
        i = torch.argsort(position).to(torch.float32)
        M = len(position)
        return torch.mean(torch.pow(position - i / (M-1),2))

    def descloss(self, DA, DB, G):
        c, h, w = DA.shape
        C = G <= 8
        C_ = G > 8
        # reshape_DA size = M, 256; reshape_DB size = 256, M
        AB = torch.matmul(DA.reshape((c,-1)).permute(1,0), DB.reshape((c,-1)))
        AB[C] = self.d * (self.m_p - AB[C])
        AB[C_] -= self.m_n
        return torch.mean(torch.clamp(AB, min=0))

    def decorrloss(self, DA, DB):
        c, h, w = DA.shape
        # reshape_DA size = 256, M
        reshape_DA = DA.reshape((c,-1))
        # reshape_DB size = 256, M
        reshape_DB = DB.reshape((c,-1))
        loss = 0
        loss += self.get_R_b(reshape_DA)
        loss += self.get_R_b(reshape_DB)
        return loss
    
    def get_R_b(self, reshape_D):
        F = reshape_D.shape[0]
        v_ = torch.mean(reshape_D, dim = 1, keepdim=True)
        V_v = reshape_D - v_
        molecular = torch.matmul(V_v, V_v.transpose(1,0))
        V_v_2 = torch.sum(torch.pow(V_v, 2), dim=1, keepdim=True)
        denominator = torch.sqrt(torch.matmul(V_v_2, V_v_2.transpose(1,0)))
        one = torch.eye(F).to(self.dev)
        return torch.sum(molecular / denominator - one) / (F * (F-1))

    def predict(self, srcipath, transformpath):
        #bath = 1
        srcimg = cv2.imread(srcipath)
        srcimg_copy = Image.fromarray(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        transformimg = Image.open(transformpath)
        srcimg_copy = transform_test(srcimg_copy)
        transformimg = transform_test(transformimg)

        srcimg_copy = torch.unsqueeze(srcimg_copy, 0)
        transformimg = torch.unsqueeze(transformimg, 0)

        srcimg_copy = srcimg_copy.to(self.dev)
        transformimg = transformimg.to(self.dev)
        As,Ap,Ad = self.forward(srcimg_copy)
        Bs,Bp,Bd = self.forward(transformimg)

        h,mask = self.get_homography(Ap[0], Ad[0], Bp[0], Bd[0], As[0],Bs[0])
        im1Reg = cv2.warpPerspective(srcimg, h, (self.w, self.h))
        print(h)
        cv2.imwrite('pre.jpg',im1Reg)
        
    def get_homography(self, Ap, Ad, Bp, Bd, As, Bs):
        Amap = self.get_position(Ap)
        Bmap = self.get_position(Bp)

        points1, points2 = self.get_match_point(Amap, Ad, Bmap, Bd, As, Bs)
        srcpath = '/home/sinjeong/gitpjt/pytorch-superpoint/datasets/HPatches/v_maskedman/1.ppm'
        transformpath = '/home/sinjeong/gitpjt/pytorch-superpoint/datasets/HPatches/v_maskedman/4.ppm'
        img = cv2.imread(srcpath)
        img_dst = cv2.imread(transformpath)
        
        map = points1
        map_dst = points2
        point_size = 1
        def random_color():
            return (random.randint(0,255), random.randint(0,255),random.randint(0,255))
        # point_color = (0, 0, 255) # BGR
        thickness = 4 # 0 、4、8
        print(len(map))

        # points to be visualized
        points_list = [(int(map[i,0]),int(map[i,1])) for i in range(len(map))]
        points_list_dst = [(int(map_dst[i,0]),int(map_dst[i,1])) for i in range(len(map))]
        
        for i, point in enumerate(points_list):
            color = random_color()
            cv2.circle(img , point, point_size, color, thickness)
            cv2.circle(img_dst , points_list_dst[i], point_size, color, thickness)
                   
        cv2.imwrite('visualization_src.jpg',img)
        
#        img = cv2.imread(srcpath)
#        map = points1
#        points_list = [(int(map[i,0]),int(map[i,1])) for i in range(len(map))]
#        print(points_list)
#        for point in points_list:
#            cv2.circle(img , point, point_size, point_color, thickness)
        cv2.imwrite('visualization_dst.jpg',img_dst)
        # print(points1)
        # print(points2)
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        return h,mask
    
    def get_match_point(self, Amap, Ad, Bmap, Bd, As, Bs):
        c = Amap.shape[0]
        c_d = Ad.shape[0]
        print(c,c_d)        
        reshape_As = As.reshape((-1)) 
        reshape_Bs = Bs.reshape((-1))
        reshape_Ap = Amap.reshape((c,-1)).permute(1,0)
        reshape_Bp = Bmap.reshape((c,-1)).permute(1,0)
        reshape_Ad = Ad.reshape((c_d,-1)).permute(1,0)
        reshape_Bd = Bd.reshape((c_d,-1))
        print(reshape_Ad.shape)
        D = torch.matmul(reshape_Ad,reshape_Bd)
        # print(D)
        A2B_nearest_Id = torch.argmax(D, dim=1)
        B2A_nearest_Id = torch.argmax(D, dim=0)

        print(A2B_nearest_Id)
        print(A2B_nearest_Id.shape)
        print(B2A_nearest_Id)
        print(B2A_nearest_Id.shape)

        match_B2A = B2A_nearest_Id[A2B_nearest_Id]
        A2B_Id = torch.from_numpy(np.array(range(len(A2B_nearest_Id)))).to(self.dev)

        print(match_B2A)
        print(match_B2A.shape)
        # for i in range(len(match_B2A)):
        #     print(match_B2A[i],end=' ')
        print(A2B_Id)
        print(A2B_Id.shape)

        finish_Id = A2B_Id == match_B2A
      
        points1 = reshape_Ap[finish_Id]
        points2 = reshape_Bp[A2B_nearest_Id[finish_Id]]

        return points1.cpu().numpy(), points2.cpu().numpy()

        # Id = torch.zeros_like(A2B_nearest, dtype=torch.uint8)
        # for i in range(len(A2B_nearest)):

def simple_train(config, output_dir, args):
    batch_size = config['training']['batch_size_train']
    epochs = config['training']['epoch_train']
    learning_rate = config['training']['learning_rate']
    savepath = os.path.join(output_dir, 'checkpoints')
    os.makedirs(savepath, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Prepare for data loader
    dataset = Picture(config, transform)
    trainloader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=config['training']['workers_train'],
                        shuffle=True, drop_last=True)

    # Prepare for model
    model = UnSuperPoint(config)
    model.train()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model.to(dev)

    # Do optimization
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    whole_step = 0
    try:
        for epoch in range(1,epochs+1):
            error = 0
            for batch_idx, (img0, img1, mat) in enumerate(trainloader):
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

                print('Train Epoch: {} [{}/{} ]\t Loss: {:.6f}'.format(epoch, batch_idx * len(img0), len(trainloader.dataset),error))
                
                if whole_step % config['save_interval'] == 0:
                    torch.save(model.state_dict(), os.path.join(save_path, config['model']['name'] + '_{}.pkl'.format(whole_step)))
                
                if whole_step % config['validation_interval'] == 0:
                    # TODO: Validation code should be implemented
                    pass

                error = 0

        torch.save(model.state_dict(), os.path.join(save_path, config['model']['name'] + '_{}.pkl'.format(whole_step)))

    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")
        torch.save(model.state_dict(), os.path.join(save_path, config['model']['name'] + '_{}.pkl'.format(whole_step)))
        pass

def simple_test(config, output_dir, args):
    model = UnSuperPoint()
    model.load_state_dict(torch.load('/home/sinjeong/unsuperpoint_allre9.pkl'))
    model.to(model.dev)
    model.train(False)
    with torch.no_grad():
        srcpath = '/home/sinjeong/gitpjt/pytorch-superpoint/datasets/HPatches/v_maskedman/1.ppm'
        transformpath = '/home/sinjeong/gitpjt/pytorch-superpoint/datasets/HPatches/v_maskedman/4.ppm'
        model.predict(srcpath, transformpath)

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
    p_test.set_defaults(func=simple_test)
    
    args = parser.parse_args()

    output_dir = os.path.join(EXPORT_PATH, args.export_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    args.func(config, output_dir, args)
    
