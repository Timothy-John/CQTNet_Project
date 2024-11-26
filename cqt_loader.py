import os,sys
from torchvision import transforms
import torch, torch.utils
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import bisect
import torchvision
import PIL

def cut_data(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = np.random.randint(max_offset)
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
    return data

def cut_data_front(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = 0
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
    return data

def shorter(feature, mean_size=2):
    length, height  = feature.shape
    new_f = np.zeros((int(length/mean_size),height),dtype=np.float64)
    for i in range(int(length/mean_size)):
        new_f[i,:] = feature[i*mean_size:(i+1)*mean_size,:].mean(axis=0)
    return new_f

def change_speed(data, l=0.7, r=1.5): # change data.shape[0]
    new_len = int(data.shape[0]*np.random.uniform(l,r))
    maxx = np.max(data)+1
    data0 = PIL.Image.fromarray((data*255.0/maxx).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize(size=(new_len,data.shape[1])), 
    ])
    new_data = transform(data0)
    return np.array(new_data)/255.0*maxx

def SpecAugment(data):
    F = 24
    f = np.random.randint(F)
    f0 = np.random.randint(84-f)
    data[f0:f0+f,:]*=0
    return data

class CQT(Dataset):
    def __init__(self, mode='train', out_length=None):
        self.indir = 'data/youtube_cqt_npy/'
        self.mode=mode
        if mode == 'train': 
            filepath='data/SHS100K-TRAIN_6'
        elif mode == 'val':
            filepath='data/SHS100K-VAL'
        elif mode == 'songs350': 
            self.indir = 'data/you350_cqt_npy/'
            filepath='data/you350_list.txt'
        elif mode == 'test': 
            filepath='data/SHS100K-TEST'
        elif mode == 'songs80': 
            self.indir = 'data/covers80_cqt_npy/'
            filepath = 'data/songs80_list.txt'
        elif mode == 'Mazurkas':
            self.indir = 'data/Mazurkas_cqt_npy/'
            filepath = 'data/Mazurkas_list.txt'
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length
    def __getitem__(self, index):
        transform_train = transforms.Compose([
            lambda x: SpecAugment(x), #SpecAugment 频谱增强一次
            lambda x: SpecAugment(x), #SpecAugment 频谱增强 x 2
            lambda x : x.T,
            lambda x : change_speed(x, 0.7, 1.3), # 速度随机变化
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        transform_test = transforms.Compose([
            lambda x : x.T,
            #lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data_front(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        filename = self.file_list[index*2].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        for i in [1,0]:
            if i==1:
                in_path1 = self.indir+str(set_id)+'_'+'0.npy'
                in_path2 = self.indir+str(set_id)+'_'+'1.npy'
            else:
                in_path1 = self.indir+str(set_id)+'_'+'0.npy'
                if index<=(len(self.file_list)/2)-6:
                    in_path2 = self.indir+str(set_id+5)+'_'+'1.npy'
                else:
                    in_path2 = self.indir+str(set_id-(len(self.file_list)/2)+5)+'_'+'1.npy'
            data1 = np.load(in_path1) # from 12xN to Nx12
            data2 = np.load(in_path2)

            if self.mode == 'train':
                data1 = transform_train(data1)
                data2 = transform_train(data2)
            else:
                data1 = transform_test(data1)
                data2 = transform_test(data2)
            yield [data1, data2], i
    def __len__(self):
        return len(self.file_list)/2

    
if __name__=='__main__':
    train_dataset = HPCP('train', 394)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
