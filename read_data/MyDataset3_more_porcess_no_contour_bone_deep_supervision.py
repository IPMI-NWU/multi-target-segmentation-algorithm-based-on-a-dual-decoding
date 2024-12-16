from torch.utils.data import Dataset
import torch
import torch.utils.data as Data
import scipy.io as scio
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import os
import pdb
import numpy as np
from transform_my_mask_no_contour import transform_rotate, transform_translate_horizontal, transform_translate_vertical, transform_flip, transform_shear

class RandomGaussianBlur(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img


def default_loader(path, IorM='rgb'):
    if IorM == 'rgb':
        return Image.open(path).convert('RGB')
    else:
        return Image.open(path).convert('L')


'''
    img_label_txt: 存储img和label的txt文件，其中第一个为原图，第二个为所有骨，第三个为后肋，第四个为前肋
'''

class MyDataset(Dataset):
    def __init__(self, img_label_txt, loader=default_loader, mode='test'):
        # print(imgtxt)
        img_label = []
        path = open(img_label_txt, 'r')
        for line in path:
            line = line.strip('\n')
            line = line.rstrip()
            img_label.append(line)

        self.img_label = img_label
        self.imgs_num = len(img_label) # 记住共有多少个文件名

        self.toTensor = transforms.ToTensor()
        self.HorizontalFlip = transforms.RandomHorizontalFlip(p=0.5)

        self.degrees = random.uniform(0, 10)
        # self.RandomAffine_degrees = transforms.RandomAffine(degrees=self.degrees)  # 转

        self.ColorJitter = transforms.ColorJitter(brightness=0.1)
        self.RandomGaussianBlur = RandomGaussianBlur()

        self.resize = transforms.Resize((512, 512))
        self.resize_map3 = transforms.Resize((512 // 2, 512 // 2))
        self.resize_map2 = transforms.Resize((512 // 4, 512 // 4))
        self.resize_map1 = transforms.Resize((512 // 8, 512 // 8))

        self.loader = loader
        self.mode = mode

        # self.RandomGaussianBlur = tr.RandomGaussianBlur()


    def __getitem__(self, index):

        # imgname = self.imgs[index]
        imglabel = self.img_label[index]

        # 0 原图的存放路径
        # 1 所有骨的存放路径
        # 2 锁骨的存放路径
        # 3 后肋的存放路径
        # 4 前肋的存放路径

        # 将路径进行分割
        temp = imglabel.strip().split('\t')
        # print(index)
        # print(temp)

        # 原图片
        img = self.loader(temp[0], IorM='rgb')  # L

        # 先改变大小
        w, h = img.size
        # 确定 w, h 对应的最终值
        for i in range(1, len(temp)):
            temp_mask = self.loader(temp[i], IorM='L')
            w_m, h_m = temp_mask.size
            w = w if w < w_m else w_m
            h = h if h < h_m else h_m

        img = img.resize((w, h))
        mask = []
        for i in range(1, len(temp)):
            temp_mask = self.loader(temp[i], IorM='Binary')
            mask.append(temp_mask.resize((w, h)))

        # print(os.path.join(self.img_path[index//self.imgs_num], imgname))
        # print(os.path.join(self.label_path[0], imgname))

        if self.mode == 'train':
            rand = random.random()
            if random.random() > 0.5:  # 水平翻转
                img, mask = transform_flip(img, mask)

            # 平移
            if random.random() < 0.25:
                img, mask = transform_translate_horizontal(img, mask, scale=random.uniform(0, 0.05))
            if random.random() < 0.25:
                img, mask = transform_translate_vertical(img, mask,  scale=random.uniform(0, 0.05))

            # 旋转
            if random.random() < 0.25:
                img, mask = transform_rotate(img, mask)

            # 错切
            if random.random() < 0.25:
                img, mask = transform_shear(img, mask)

            if random.random() >= 0.6:  # 对比度变换
                img = self.ColorJitter(img)

            # img = self.RandomGaussianBlur(img)

        img = self.resize(img)
        img = self.toTensor(img)  # (1, 512, 512)

        mask_map3 = []
        mask_map2 = []
        mask_map1 = []
        for i in range(len(mask)):
            mask[i] = self.resize(mask[i])
            mask_map3.append(self.resize_map3(mask[i]))
            mask_map2.append(self.resize_map2(mask[i]))
            mask_map1.append(self.resize_map1(mask[i]))


            mask[i] = self.toTensor(mask[i])
            mask_map3[i] = self.toTensor(mask_map3[i])
            mask_map2[i] = self.toTensor(mask_map2[i])
            mask_map1[i] = self.toTensor(mask_map1[i])

        return img, mask_map1, mask_map2, mask_map3, mask, temp[0]

    def __len__(self):
        return len(self.img_label)  # 与index相关的



if __name__ == '__main__':
    img_label_txt = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/txt/test.txt'

    train_datasets = MyDataset(img_label_txt, mode='test')
    trainloader = Data.DataLoader(dataset=train_datasets, batch_size=1, shuffle=False, num_workers=0)

    for step, (imgs, mask, _) in enumerate(trainloader):
        # print(mask[0].shape)
        # print(imgs.shape)
        print(len(mask))



