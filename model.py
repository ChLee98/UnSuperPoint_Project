import os
import random

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from settings import DEFAULT_SETTING
from utils.utils import batch_warp_points, normPts, denormPts

# TEMP
import torchvision.transforms as transforms

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[1/0.225,1/0.225,1/0.225])
])
# -----

class UnSuperPoint(nn.Module):
    def __init__(self, config=None):
        super(UnSuperPoint, self).__init__()
        self.config = config
        if not config:
            self.config = DEFAULT_SETTING
        self.usp = self.config['model']['usp_loss']['alpha_usp']
        self.position_weight = self.config['model']['usp_loss']['alpha_position']
        self.score_weight = self.config['model']['usp_loss']['alpha_score']
        self.uni_xy = self.config['model']['unixy_loss']['alpha_unixy']
        self.desc = self.config['model']['desc_loss']['alpha_desc']
        self.d = self.config['model']['desc_loss']['lambda_d']
        self.m_p = self.config['model']['desc_loss']['margin_positive']
        self.m_n = self.config['model']['desc_loss']['margin_negative']
        self.decorr = self.config['model']['decorr_loss']['alpha_decorr']
        self.correspond = self.config['model']['correspondence_threshold']
        self.conf_thresh = self.config['model']['detection_threshold']
        self.nn_thresh = self.config['model']['nn_thresh']

        self.border_remove = 4  # Remove points this close to the border.
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

    def tb_add_loss(self, loss, task='train'):
        for k in loss:
            self.writer.add_scalar('{}/{}'.format(task, k), loss[k].item(), self.step)

    @torch.no_grad()
    def tb_add_hist(self, name, ts, bin=100, task='train'):
        f = plt.figure()
        plt.hist(ts.reshape(-1).cpu().numpy(), bin, density=True)
        plt.close()
        self.writer.add_figure('{}/{}'.format(task, name), f, self.step)

    def train_val_step(self, img0, img1, mat, task='train'):
        self.task = task
        #with torch.autograd.detect_anomaly():
        img0 = img0.to(self.dev)
        img1 = img1.to(self.dev)
        mat = mat.squeeze()
        mat = mat.to(self.dev)
        self.optimizer.zero_grad()
        
        if task == 'train':
            s1,p1,d1 = self.forward(img0)
            s2,p2,d2 = self.forward(img1)

        else:
            with torch.no_grad():
                s1,p1,d1 = self.forward(img0)
                s2,p2,d2 = self.forward(img1)

        lossdict = self.loss(s1,p1,d1,s2,p2,d2,mat)
        self.tb_add_loss(lossdict, task)

        if task == 'train' and self.step % self.config['tensorboard_interval'] == 0:
            self.tb_add_hist('train/left/x_relative', p1[0])
            self.tb_add_hist('train/left/y_relative', p1[1])
            self.tb_add_hist('train/left/score', s1)
            self.tb_add_hist('train/right/x_relative', p2[0])
            self.tb_add_hist('train/right/y_relative', p2[1])
            self.tb_add_hist('train/right/score', s2)

        if task == 'valid':
            self.tb_add_hist('valid/left/x_relative', p1[0])
            self.tb_add_hist('valid/left/y_relative', p1[1])
            self.tb_add_hist('valid/left/score', s1)
            self.tb_add_hist('valid/right/x_relative', p2[0])
            self.tb_add_hist('valid/right/y_relative', p2[1])
            self.tb_add_hist('valid/right/score', s2)
            # TODO: Validation operations

        if task == 'train':
            lossdict['loss'].backward()
            self.optimizer.step()

        return lossdict['loss'].item()

    def loss(self, bath_As, bath_Ap, bath_Ad, 
        bath_Bs, bath_Bp, bath_Bd, mat):
        usp = 0; unixy = 0; desc = 0; decorr = 0
        usp, unixy, desc, decorr = self.UnSuperPointLoss(bath_As, bath_Ap, bath_Ad, bath_Bs, bath_Bp, bath_Bd, mat)

        loss = usp + unixy + desc + decorr
        lossdict = {
            "loss": loss,
            "usp_loss": usp,
            "uni_xy_loss": unixy,
            "descriptor_loss": desc,
            "decorrelation_loss": decorr
        }
        
        return lossdict

    def UnSuperPointLoss(self, As, Ap, Ad, Bs, Bp, Bd, mat):
        position_A = self.get_position(Ap, mat=mat)
        position_B = self.get_position(Bp)
        G = self.getG(position_A, position_B)

        Usploss = self.usploss(As, Bs, mat, G)
        Uni_xyloss = self.uni_xyloss(Ap, Bp)

        Descloss = self.descloss(Ad, Bd, G)
        Decorrloss = self.decorrloss(Ad, Bd)
        return self.usp * Usploss, self.uni_xy * Uni_xyloss,\
            self.desc * Descloss, self.decorr * Decorrloss

    def get_position(self, Pmap, mat=None):
        x = 0
        y = 1
        res = torch.zeros_like(Pmap)
        # print(Pmap.shape,res.shape)
        for i in range(Pmap.shape[3]):
            res[:,x,:,i] = (i + Pmap[:,x,:,i]) * self.downsample
        for i in range(Pmap.shape[2]):
            res[:,y,i,:] = (i + Pmap[:,y,i,:]) * self.downsample 
        if mat is not None:
            # print(mat.shape)
            if self.task == 'train':
                shape = torch.tensor([Pmap.shape[3], Pmap.shape[2]]).to(self.dev) * self.downsample
                B = Pmap.shape[0]
                Hc, Wc = Pmap.shape[2:]
                res = normPts(res.permute(0, 2, 3, 1).reshape((B, -1, 2)), shape)
                # r = torch.stack((res[:,1], res[:,0]), dim=1) # (y, x) to (x, y)
                r = batch_warp_points(res, mat, self.dev)
                # r = torch.stack((r[:,1], r[:,0]), dim=1)  # (x, y) to (y, x)
                r = denormPts(r, shape).reshape(B, Hc, Wc, 2).permute(0, 3, 1, 2)
            else:
                r = torch.zeros_like(res)
                Denominator = res[:,x,:,:]*(mat[:,2,0].reshape(-1, 1, 1)) + res[:,y,:,:]*(mat[:,2,1].reshape(-1, 1, 1)) +(mat[:,2,2].reshape(-1, 1, 1))
                r[:,x,:,:] = (res[:,x,:,:]*(mat[:,0,0].reshape(-1, 1, 1)) + 
                    res[:,y,:,:]*(mat[:,0,1].reshape(-1, 1, 1)) + (mat[:,0,2].reshape(-1, 1, 1))) / (Denominator + 1e-8)
                r[:,y,:,:] = (res[:,x,:,:]*(mat[:,1,0].reshape(-1, 1, 1)) + 
                    res[:,y,:,:]*(mat[:,1,1].reshape(-1, 1, 1)) +(mat[:,1,2].reshape(-1, 1, 1))) / (Denominator + 1e-8)
            return r
        else:
            return res

    def usploss(self, As, Bs, mat, G):
        A2BId, Id = self.get_point_pair(G, As, Bs)
        # print(A2BId.shape, Id.shape)
        # print(reshape_As_k.shape,reshape_Bs_k.shape,d_k.shape)
        B = G.shape[0]
        reshape_As = As.reshape(B, -1)
        reshape_Bs = Bs.reshape(B, -1)
        positionK_loss = scoreK_loss = uspK_loss = 0

        for i in range(B):
            d_k = G[i][Id[i], A2BId[i][Id[i]]]
            reshape_As_k = reshape_As[i][Id[i]]
            reshape_Bs_k = reshape_Bs[i][A2BId[i][Id[i]]]
            positionK_loss += torch.mean(d_k)
            scoreK_loss += torch.mean(torch.pow(reshape_As_k - reshape_Bs_k, 2))
            uspK_loss += self.get_uspK_loss(d_k, reshape_As_k, reshape_Bs_k)

        return (self.position_weight * positionK_loss + 
            self.score_weight * scoreK_loss + uspK_loss)/B

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

    def getG(self, PA, PB):
        b, c = PA.shape[0:2]
        # reshape_PA shape = b, 2, m -> b, m, 2
        reshape_PA = PA.reshape((b, c, -1)).permute(0, 2, 1)
        # reshape_PB shape = b, 2, m -> b, m, 2
        reshape_PB = PB.reshape((b, c, -1)).permute(0, 2, 1)
        # x shape b,m,m <- b, (m,1 - 1,m) 
        x = torch.unsqueeze(reshape_PA[:,:,0],2) - torch.unsqueeze(reshape_PB[:,:,0],1)
        # y shape b,m,m <- b, (m,1 - 1,m)
        y = torch.unsqueeze(reshape_PA[:,:,1],2) - torch.unsqueeze(reshape_PB[:,:,1],1)

        G = torch.sqrt(torch.pow(x,2) + torch.pow(y,2))

        return G

    def get_point_pair(self, G, As, Bs):
        A2B_min_Id = torch.argmin(G, dim=2)
        B, M = A2B_min_Id.shape[0:2]
        Id = torch.nn.functional.grid_sample(G.unsqueeze(1), 
            torch.stack([torch.arange(M).reshape(1, M).repeat(B, 1).to(self.dev), A2B_min_Id],
                dim=2).unsqueeze(2).float())
        Id = Id.squeeze() <= self.correspond
        return A2B_min_Id, Id

    def get_uspK_loss(self, d_k, reshape_As_k, reshape_Bs_k):
        sk_ = (reshape_As_k + reshape_Bs_k) / 2
        d_ = torch.mean(d_k)
        return torch.mean(sk_ * (d_k - d_))

    def uni_xyloss(self, Ap, Bp):
        b, c = Ap.shape[0:2]
        reshape_PA = Ap.reshape((b,c,-1)).transpose(2,1)
        reshape_PB = Bp.reshape((b,c,-1)).transpose(2,1)
        loss = 0
        for i in range(2):
            loss += self.get_uni_xy(reshape_PA[:,:,i], b)
            loss += self.get_uni_xy(reshape_PB[:,:,i], b)
        return loss
        
    def get_uni_xy(self, position, batch):
        pos, _ = torch.sort(position)
        M = position.shape[1]
        i = torch.arange(0., M, requires_grad=False).to(self.dev)
        return torch.mean(torch.pow(pos - i.unsqueeze(0) / (M-1),2))

    def descloss(self, DA, DB, G):
        b, c, h, w = DA.shape
        C = G <= 8
        C_ = G > 8
        # reshape_DA size = M, 256; reshape_DB size = 256, M
        AB = torch.bmm(DA.reshape((b, c,-1)).transpose(2,1), DB.reshape((b, c,-1)))
        AB[C] = self.d * (self.m_p - AB[C])
        AB[C_] -= self.m_n
        return torch.mean(torch.clamp(AB, min=0))

    def decorrloss(self, DA, DB):
        b, c, h, w = DA.shape
        # reshape_DA size = 256, M
        reshape_DA = DA.reshape((b,c,-1))
        # reshape_DB size = 256, M
        reshape_DB = DB.reshape((b,c,-1))
        loss = 0
        loss += self.get_R_b(reshape_DA)
        loss += self.get_R_b(reshape_DB)
        return loss
    
    def get_R_b(self, reshape_D):
        B, F = reshape_D.shape[0:2]
        v_ = torch.mean(reshape_D, dim = 2, keepdim=True)
        V_v = reshape_D - v_
        molecular = torch.matmul(V_v, V_v.transpose(2,1))
        V_v_2 = torch.sum(torch.pow(V_v, 2), dim=2, keepdim=True)
        denominator = torch.sqrt(torch.matmul(V_v_2, V_v_2.transpose(2,1)))
        one = torch.eye(F).to(self.dev).unsqueeze(0)
        if torch.isnan(molecular).any() or torch.isnan(denominator).any():
            print("NAN DETECTED previous!!!")
            print("B, F : ", B, F)
        rb = torch.sum(torch.square((molecular / (denominator + 1e-8) - one) / (B * F * (F-1))))
        if torch.isnan(rb).any():
            print("NAN DETECTED rb!!")
        return rb

    def getPtsDescFromHeatmap(self, point, heatmap, desc):
        '''
        :param self:
        :param point:
            np (2, Hc, Wc)
        :param heatmap:
            np (Hc, Wc)
        :param desc:
            np (256, Hc, Wc)
        :return:
        '''
        heatmap = heatmap.squeeze()
        desc = desc.squeeze()
        # print("heatmap sq:", heatmap.shape)
        H = heatmap.shape[0]*self.downsample
        W = heatmap.shape[1]*self.downsample
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = point[0, xs, ys] # abuse of ys, xs
        pts[1, :] = point[1, xs, ys]
        pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
        desc = desc[:, xs, ys]

        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        desc = desc[:, inds[::-1]]

        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        desc = desc[:, ~toremove]
        return pts[:, :300], desc[:, :300]

    # TODO: No more need; eliminate this part
    def predict(self, srcipath, transformpath, output_dir):
        srcimg = cv2.imread(srcipath)
        srcimg_copy = Image.fromarray(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        transformimg = cv2.imread(transformpath)
        transformimg_copy = Image.fromarray(cv2.cvtColor(transformimg, cv2.COLOR_BGR2RGB))
        srcimg_copy = transform_test(srcimg_copy)
        transformimg_copy = transform_test(transformimg_copy)

        srcimg_copy = torch.unsqueeze(srcimg_copy, 0)
        transformimg_copy = torch.unsqueeze(transformimg_copy, 0)

        srcimg_copy = srcimg_copy.to(self.dev)
        transformimg_copy = transformimg_copy.to(self.dev)
        As,Ap,Ad = self.forward(srcimg_copy)
        Bs,Bp,Bd = self.forward(transformimg_copy)

        h, mask, points1, points2 = self.get_homography(Ap[0], Ad[0], Bp[0], Bd[0], As[0],Bs[0])
        im1Reg = cv2.warpPerspective(srcimg, h, (self.w, self.h))
        print(h)

        # Visualize the points
        point_size = 1
        thickness = 4 # 0, 4, 8
        def random_color():
            return (random.randint(0,255), random.randint(0,255),random.randint(0,255))

        points_list = [(int(points1[i,0]),int(points1[i,1])) for i in range(len(points1))]
        points_list_dst = [(int(points2[i,0]),int(points2[i,1])) for i in range(len(points2))]
        for i, point in enumerate(points_list):
            color = random_color()
            cv2.circle(srcimg , point, point_size, color, thickness)
            cv2.circle(transformimg , points_list_dst[i], point_size, color, thickness)

        cv2.imwrite(os.path.join(output_dir, 'visualization_src.jpg'), srcimg)
        cv2.imwrite(os.path.join(output_dir, 'visualization_dst.jpg'), transformimg)
        cv2.imwrite(os.path.join(output_dir, 'visualization_s2d.jpg'), im1Reg)
        
    def get_homography(self, Ap, Ad, Bp, Bd, As, Bs):
        Amap = self.get_position(Ap)
        Bmap = self.get_position(Bp)

        points1, points2 = self.get_match_point(Amap, Ad, Bmap, Bd, As, Bs)
        # img = cv2.imread(srcpath)
        # map = points1
        # points_list = [(int(map[i,0]),int(map[i,1])) for i in range(len(map))]
        # print(points_list)
        # for point in points_list:
        #     cv2.circle(img , point, point_size, point_color, thickness)
        # print(points1)
        # print(points2)
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        return h, mask, points1, points2
    
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
