import os
import torch
from cqt_loader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from config import opt
from torchnet import meter
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from utility import *

def custom_collate(batch):
    # Separate data and labels
    data1 = [item[0][0] for item in batch]
    data2 = [item[0][1] for item in batch]
    labels = [item[1] for item in batch]

    # Stack the data tensors
    data1 = torch.stack(data1)
    data2 = torch.stack(data2)
    
    # Convert labels to tensor
    labels = torch.LongTensor(labels)
    return [data1, data2], labels

# multi_size train
def multi_train(**kwargs):
    parallel = True 
    opt.model = 'CQTNet'
    opt.notes='CQTNet'
    opt.batch_size=32
    #opt.load_latest=True
    #opt.load_model_path = ''
    opt._parse(kwargs)
    # step1: configure model
    
    model = getattr(models, opt.model)() 
    if parallel is True: 
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    model.to(opt.device)
    print(model)
    # step2: data
   
    train_data0_1 = CQT('train', label=1, out_length=200)
    train_data1_1 = CQT('train', label=1, out_length=300)
    train_data2_1 = CQT('train', label=1, out_length=400)
    
    train_data0_0 = CQT('train', label=0, out_length=200)
    train_data1_0 = CQT('train', label=0, out_length=300)
    train_data2_0 = CQT('train', label=0, out_length=400)
    # val_data350 = CQT('songs350', out_length=None)
    val_data80_1 = CQT('songs80', label=1, out_length=None)
    val_data_1 = CQT('val', label=1, out_length=None)
    test_data_1 = CQT('test', label=1, out_length=None)
    
    val_data80_0 = CQT('songs80', label=0, out_length=None)
    val_data_0 = CQT('val', label=0, out_length=None)
    test_data_0 = CQT('test', label=0, out_length=None)
    # val_datatMazurkas = CQT('Mazurkas', out_length=None)
    ## train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    ## train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    ## train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    ## val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
    ## test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
    ## val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    train_dataloader0_1 = DataLoader(train_data0_1, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    train_dataloader1_1 = DataLoader(train_data1_1, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    train_dataloader2_1 = DataLoader(train_data2_1, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_dataloader_1 = DataLoader(val_data_1, 1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_dataloader_1 = DataLoader(test_data_1, 1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    val_dataloader80_1 = DataLoader(val_data80_1, 1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    
    train_dataloader0_0 = DataLoader(train_data0_0, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    train_dataloader1_0 = DataLoader(train_data1_0, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    train_dataloader2_0 = DataLoader(train_data2_0, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_dataloader_0 = DataLoader(val_data_0, 1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_dataloader_0 = DataLoader(test_data_0, 1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    val_dataloader80_0 = DataLoader(val_data80_0, 1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    # val_dataloader350 = DataLoader(val_data350, 1, shuffle=False, num_workers=1)
    # val_dataloaderMazurkas = DataLoader(val_datatMazurkas,1, shuffle=False,num_workers=1)
    #step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=opt.lr_decay,patience=2, verbose=True,min_lr=5e-6)
    #train label 1
    best_MAP=0
    #val_slow(model, val_dataloader80_1, -1)
    for epoch in range(opt.max_epoch):
        running_loss = 0
        num = 0
        for (data0, label0),(data1, label1),(data2, label2) in tqdm(zip(train_dataloader0_1, train_dataloader1_1, train_dataloader2_1)):
            for flag in range(3):
                if flag==0:
                    data=data0
                    label=label0
                elif flag==1:
                    data=data1
                    label=label1
                else:
                    data=data2
                    label=label2
                # train model
                input1 = data[0].requires_grad_()
                input2 = data[1].requires_grad_()
                input1 = input1.to(opt.device)
                input2 = input2.to(opt.device)
                target = label.to(opt.device)

                optimizer.zero_grad()
                score, _ = model([input1, input2])
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num += target.shape[0]
        running_loss /= num 
        print(running_loss)
        if parallel is True:
            model.module.save(opt.notes)
        else:
            model.save(opt.notes)
        # update learning rate
        scheduler.step(running_loss) 
        # validate
        MAP=0
        #MAP += val_slow(model, val_dataloader350, epoch)
        MAP += val_slow(model, val_dataloader80_1, epoch)
        #val_slow(model, val_dataloaderMazurkas, epoch)
        if MAP>best_MAP:
            best_MAP=MAP
            print('*****************BEST*****************')
        print('')
        model.train()
    
    #train label 0
    best_MAP=0
    val_slow(model, val_dataloader80_0, -1)
    for epoch in range(opt.max_epoch):
        running_loss = 0
        num = 0
        for (data0, label0),(data1, label1),(data2, label2) in tqdm(zip(train_dataloader0_0, train_dataloader1_0, train_dataloader2_0)):
            for flag in range(3):
                if flag==0:
                    data=data0
                    label=label0
                elif flag==1:
                    data=data1
                    label=label1
                else:
                    data=data2
                    label=label2
                # train model
                input1 = data[0].requires_grad_()
                input2 = data[1].requires_grad_()
                input1 = input1.to(opt.device)
                input2 = input2.to(opt.device)
                target = label.to(opt.device)

                optimizer.zero_grad()
                score, _ = model([input1, input2])
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num += target.shape[0]
        running_loss /= num 
        print(running_loss)
        if parallel is True:
            model.module.save(opt.notes)
        else:
            model.save(opt.notes)
        # update learning rate
        scheduler.step(running_loss) 
        # validate
        MAP=0
        #MAP += val_slow(model, val_dataloader350, epoch)
        MAP += val_slow(model, val_dataloader80_0, epoch)
        #val_slow(model, val_dataloaderMazurkas, epoch)
        if MAP>best_MAP:
            best_MAP=MAP
            print('*****************BEST*****************')
        print('')
        model.train()

   
@torch.no_grad()
def multi_val_slow(model, dataloader1,dataloader2, epoch):
    model.eval()
    labels, features,features2 = None, None, None
    for ii, (data, label) in enumerate(dataloader1):
        input = data.to(opt.device)
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
    for ii, (data, label) in enumerate(dataloader2):
        input = data.to(opt.device)
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        if features2 is not None:
            features2 = np.concatenate((features2, feature), axis=0)
        else:
            features2 = feature
            
    features = norm(features+features2)
    dis2d = get_dis2d4(features)
    
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels,[100, 350])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)

    print(epoch, MAP, top10, rank1 )
    model.train()
    return MAP

    
@torch.no_grad()
def val_slow(model, dataloader, epoch):
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label) in enumerate(dataloader):
        input1 = data[0].to(opt.device)
        input2 = data[1].to(opt.device)
        #print(input.shape)
        score, feature = model([input1, input2])
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
    features = norm(features)
    #dis2d = get_dis2d4(features)
    dis2d = -np.matmul(features, features.T) # [-1,1] Because normalized, so mutmul is equal to ED
    np.save('dis80.npy',dis2d)
    np.save('label80.npy',labels)
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels,[100, 350])
    #elif len(labels) == 160:    MAP, top10, rank1 = calc_MAP(dis2d, labels,[80, 160])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)

    print(epoch, MAP, top10, rank1 )
    model.train()
    return MAP

def test(**kwargs):
    opt.batch_size=1
    opt.num_workers=1
    opt.model = 'CQTNet'
    opt.load_latest = False
    opt.load_model_path = 'check_points/CQTNet.pth'
    opt._parse(kwargs)
    
    model = getattr(models, opt.model)() 
    #print(model)
    if opt.load_latest is True:
        model.load_latest(opt.notes)
    elif opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    #val_data350 = CQT('songs350', out_length=None)
    #val_data80 = CQT('songs80', out_length=None)
    val_data = CQT('val', out_length=None)
    test_data = CQT('test', out_length=None)
    #val_datatMazurkas = CQT('Mazurkas', out_length=None)
    val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
    test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
    #val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    #val_dataloader350 = DataLoader(val_data350, 1, shuffle=False, num_workers=1)
    #val_dataloaderMazurkas = DataLoader(val_datatMazurkas,1, shuffle=False,num_workers=1)
    
    #val_slow(model, val_dataloader350, 0)
    val_slow(model, val_dataloader80_1, 0)
    val_slow(model, val_dataloader80_0, 0)
    #val_slow(model, val_dataloaderMazurkas, 0)



if __name__=='__main__':
    import fire
    fire.Fire()
