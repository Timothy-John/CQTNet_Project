import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from torchvision import transforms
import random
import PIL
import torch.nn.functional as F

class CQT(Dataset):
    def __init__(self, mode='train', label=None, out_length=None):
        self.label = label
        self.indir = '/content/CQTNet_Binary/data/youtube_hpcp_npy/'
        self.mode = mode
        if mode == 'train': 
            filepath = 'data/SHS100K-TRAIN_6'
        elif mode == 'val':
            filepath = 'data/SHS100K-VAL'
        elif mode == 'test': 
            filepath = 'data/SHS100K-TEST'
        elif mode == 'songs80': 
            self.indir = 'data/covers80_cqt_npy/'
            filepath = 'data/songs80_list.txt'
        
        with open(filepath, 'r') as fp:
            self.file_list = sorted([line.rstrip() for line in fp])
        self.out_length = out_length

    def pad_or_truncate(self, data, target_length, target_freq=84):
        # Ensure data is 3D: (channels, frequency, time)
        if data.ndim == 2:
            data = data.unsqueeze(0)  # Add channel dimension
        
        _, freq, current_length = data.shape
        
        # Pad or truncate frequency dimension
        if freq > target_freq:
            data = data[:, :target_freq, :]
        elif freq < target_freq:
            pad_freq = target_freq - freq
            data = F.pad(data, (0, 0, 0, pad_freq), mode='constant', value=0)
        
        # Pad or truncate time dimension
        if current_length > target_length:
            data = data[:, :, :target_length]
        elif current_length < target_length:
            pad_time = target_length - current_length
            data = F.pad(data, (0, pad_time), mode='constant', value=0)
        return data

    def __getitem__(self, index):
        transform_train = transforms.Compose([
            lambda x: self.SpecAugment(x),
            lambda x: self.SpecAugment(x),
            lambda x: self.change_speed(x, 0.7, 1.3),
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: self.cut_data(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.unsqueeze(0),  # Add channel dimension
        ])
        
        transform_test = transforms.Compose([
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: self.cut_data_front(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.unsqueeze(0),  # Add channel dimension
        ])
        
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        if self.label==1:
            in_path1 = self.indir+str(set_id)+'_'+str(version_id)+'.npy'
            if ((index+1)<len(self.file_list)) and (int(self.file_list[index+1].strip().split('.')[0].split('_')[0]) == int(set_id)):
              in_path2 = self.indir+str(set_id)+'_'+str(self.file_list[index+1].strip().split('.')[0].split('_')[1])+'.npy'
            else:
              in_path2 = self.indir+str(set_id)+'_'+'0.npy'
        if self.label==0:
            in_path1 = self.indir+str(set_id)+'_'+str(version_id)+'.npy'
            if (index+1)<len(self.file_list):
              ind = sorted(list(np.unique([line.split('_')[0] for line in self.file_list]))).index(str(set_id))
              set_id_n = sorted(list(np.unique([line.split('_')[0] for line in self.file_list])))[ind+1]
              in_path2 = self.indir+str(set_id_n)+'_'+'0.npy'
            else:
              in_path2 = self.indir+'0_0.npy'
        data1 = np.load(in_path1) # from 12xN to Nx12
        data2 = np.load(in_path2)
            
        if self.mode == 'train':
            data1 = transform_train(data1)
            data2 = transform_train(data2)
        else:
            data1 = transform_test(data1)
            data2 = transform_test(data2)

        return [data1, data2], self.label
    def __len__(self):
        return len(self.file_list)

    def SpecAugment(self, data):
        F = 24
        f = np.random.randint(F)
        f0 = np.random.randint(84 - f)
        data[f0:f0 + f, :] *= 0
        return data

    def change_speed(self, data, l=0.7, r=1.3):
        new_len = int(data.shape[1] * np.random.uniform(l, r))
        maxx = np.max(data) + 1
        data0 = PIL.Image.fromarray((data * 255.0 / maxx).astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize(size=(84, new_len)),
        ])
        new_data = transform(data0)
        return np.array(new_data) / 255.0 * maxx

    def cut_data(self, data, out_length):
        if out_length is not None:
            if data.shape[1] > out_length:
                max_offset = data.shape[1] - out_length
                offset = np.random.randint(max_offset)
                data = data[:, offset:(out_length+offset)]
            else:
                offset = out_length - data.shape[1]
                data = np.pad(data, ((0, 0), (0, offset)), "constant")
        if data.shape[1] < 200:
            offset = 200 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, offset)), "constant")
        return data

    def cut_data_front(self, data, out_length):
        if out_length is not None:
            if data.shape[1] > out_length:
                data = data[:, :out_length]
            else:
                offset = out_length - data.shape[1]
                data = np.pad(data, ((0, 0), (0, offset)), "constant")
        if data.shape[1] < 200:
            offset = 200 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, offset)), "constant")
        return data

if __name__ == '__main__':
    train_dataset = CQT('train', 394)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
