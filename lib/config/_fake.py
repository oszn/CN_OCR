from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2

class _FAKE(data.Dataset):
    def __init__(self, config, is_train=True):
        self.lbtxt=config.DATASET.LB
        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)
        self.img_list=os.listdir(self.root)
        # char_file = config.DATASET.CHAR_FILE
        # self.lable=os.listdir(self.lbroot)
        f=open(self.lbtxt,encoding="utf8")
        liens=f.readlines()
        self.labels={}
        for line in liens:
            name,content=line.split(" ")
            name=name.replace("0/","")
            self.labels[name]=content

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name=self.img_list[idx]
        img_path=os.path.join(self.root,img_name)

        # img_name = list(self.labels[idx].keys())[0]
        img = cv2.imread(os.path.join(self.root, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        return img, self.labels[img_name],idx








