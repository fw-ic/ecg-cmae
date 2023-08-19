# Author: Nabil Ibtehaz (https://github.com/nibtehaz)


import torch.nn as nn
import torch
from MAE1DCorrelated import MaskedAutoencoderViT1DCorrelated
from MAEBank import MAEBank
import numpy as np
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util.ecg_dataset import ECGPretrainDataset
import seaborn as sns
import os
import sys
sns.set()

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, num_channels, out_channels, activation='ReLU', kernel_size=3, padding=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv1d(num_channels, out_channels,
                              kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class UNet(nn.Module):
    def __init__(self, in_channels):
        '''
        in_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.in_channels = in_channels
        # Question here
        num_channels = 64


        #self.inp = ConvBatchNorm(in_channels, num_channels)


        self.pool = nn.MaxPool1d(2)


        self.cnv11 = ConvBatchNorm(in_channels   , num_channels)
        self.cnv12 = ConvBatchNorm(num_channels , num_channels)


        self.cnv21 = ConvBatchNorm(num_channels , num_channels*2)
        self.cnv22 = ConvBatchNorm(num_channels*2 , num_channels*2)


        self.cnv31 = ConvBatchNorm(num_channels*2 , num_channels*4)
        self.cnv32 = ConvBatchNorm(num_channels*4 , num_channels*4)


        self.cnv41 = ConvBatchNorm(num_channels*4 , num_channels*8)
        self.cnv42 = ConvBatchNorm(num_channels*8, num_channels*8)

        self.cnv51 = ConvBatchNorm(num_channels*8, num_channels*16)
        self.cnv52 = ConvBatchNorm(num_channels*16, num_channels*16)

        self.up6 = torch.nn.ConvTranspose1d(num_channels*16,num_channels*8,kernel_size=2,stride=2)


        self.cnv61 = ConvBatchNorm(num_channels*8 + num_channels*8, num_channels*8)
        self.cnv62 = ConvBatchNorm(num_channels*8, num_channels*8)

        self.up7 = torch.nn.ConvTranspose1d(num_channels*8,num_channels*4,kernel_size=2,stride=2)

        self.cnv71 = ConvBatchNorm(num_channels*4  + num_channels*4, num_channels*4)
        self.cnv72 = ConvBatchNorm(num_channels*4 , num_channels*4)


        self.up8 = torch.nn.ConvTranspose1d(num_channels*4,num_channels*2,kernel_size=2,stride=2)

        self.cnv81 = ConvBatchNorm(num_channels*2 + num_channels*2, num_channels*2)
        self.cnv82 = ConvBatchNorm(num_channels*2, num_channels*2)

        self.up9 = torch.nn.ConvTranspose1d(num_channels*2,num_channels,kernel_size=2,stride=2)

        self.cnv91 = ConvBatchNorm(num_channels + num_channels, num_channels)
        self.cnv92 = ConvBatchNorm(num_channels, num_channels)

        
        self.out = torch.nn.Conv1d(num_channels, 1,kernel_size=1)
        
    def forward(self, x):
        
        
        #x1 = self.inp(x)
        x1 = x

        x2 = self.cnv11(x1)
        x2 = self.cnv12(x2)
        
        x2p = self.pool(x2)

        x3 = self.cnv21(x2p)
        x3 = self.cnv22(x3)
        
        x3p = self.pool(x3)
        
        x4 = self.cnv31(x3p)
        x4 = self.cnv32(x4)
        
        x4p = self.pool(x4)
        
        x5 = self.cnv41(x4p)
        x5 = self.cnv42(x5)
        
        x5p = self.pool(x5)
        
        x6 = self.cnv51(x5p)
        x6 = self.cnv52(x6)
        
        x7 = self.up6(x6)
        
        x7 = self.cnv61(torch.cat([x7,x5],dim=1))
        x7 = self.cnv62(x7)
        
        x8 = self.up7(x7)
        
        x8 = self.cnv71(torch.cat([x8,x4],dim=1))
        x8 = self.cnv72(x8)
        
        x9 = self.up8(x8)
        
        x9 = self.cnv81(torch.cat([x9,x3,],dim=1))
        x9 = self.cnv82(x9)
        
        x10 = self.up9(x9)
        
        x10 = self.cnv91(torch.cat([x10,x2],dim=1))
        x10 = self.cnv92(x10)
        
        '''print(x1.size())
        print(x2.size())
        print(x3.size())
        print(x4.size())
        print(x5.size())
        print(x6.size())
        print(x7.size())
        print(x8.size())
        print(x9.size())
        print(x10.size())'''


        
        pred = self.out(x10)
        
        return pred

def train_unet_model(maebank_path, channl_id, gpu_mae, gpu_unet, batch_size, n_epochs, lr=1e-3):

    # channl_id is 1-indexed
    mask_ratio = 0.75
    sys.stdout = open(os.path.join(maebank_path,'logs',f'log_unet{channl_id}.log'),'w')

    mdl_bnk = MAEBank(['cpu'])

    mdl_bnk.maes[channl_id-1].load_state_dict(
        torch.load(
        f"{maebank_path}/saved_models/mae_channel_{channl_id}_best.pth",
        map_location=f"cpu",
        )   
    )

    mdl_bnk.maes[channl_id-1].to(f"cuda:{gpu_mae}")
    mdl_bnk.maes[channl_id-1].eval()

    unt_mdl = UNet(1)
    unt_mdl.to(f"cuda:{gpu_unet}")

    data_path = "/data/nabil/ecg_repr/physionet.org/files/challenge-2021/1.0.3/training/"
    dataset_train = ECGPretrainDataset(data_path,2500,mode='1D',channels='single')
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
                    dataset_train, sampler=sampler_train,
                    batch_size=batch_size,
                    num_workers=10,
                    pin_memory=True,
                    drop_last=True)

    optimizer = torch.optim.Adam(unt_mdl.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)


    unet_criterion = torch.nn.MSELoss()

    best_loss = 1e6

    for epoch in range(n_epochs):

        cur_loss = 0
    
        for data_iter_step, samples in enumerate(data_loader_train):

            with torch.no_grad():
                samples = samples.to(torch.float32)
                samples = samples.to(f"cuda:{gpu_mae}")
                pred = mdl_bnk.maes[channl_id-1].predict_reconstruction(samples,mdl_bnk.num_patches,mask_ratio)
                decoded_sigs = mdl_bnk.maes[channl_id-1].unpatchify(pred)

                samples = samples.to('cpu')
                pred = pred.to('cpu')
                decoded_sigs = decoded_sigs.to('cpu')

            optimizer.zero_grad()

            decoded_sigs = decoded_sigs.to(f"cuda:{gpu_unet}")
            samples = samples.to(f"cuda:{gpu_unet}")

            decoded_sigs = decoded_sigs[:,:,2:-2]
            samples = samples[:,:,2:-2]

            unt_pred = unt_mdl(decoded_sigs)
            unt_pred = torch.clip(unt_pred, -1, 1)

            samples = samples / 10      

            loss = unet_criterion(unt_pred,samples)
            loss.backward()

            optimizer.step()

            loss = loss.to('cpu')
            decoded_sigs = decoded_sigs.to('cpu')
            unt_pred = unt_pred.to('cpu')
            samples = samples.to('cpu')

            cur_loss += loss.item()/len(data_loader_train)

            #print(f'[{epoch}] [{data_iter_step}/{len(data_loader_train)}] : MSE loss : {loss.item()}')

        print(f'EPOCH - {epoch} : MSE loss {cur_loss}')
        scheduler.step()

        if cur_loss < best_loss:
            best_loss = cur_loss
            torch.save(unt_mdl.state_dict(), os.path.join(maebank_path,'saved_models',f'unet_channel_{channl_id}.pth'))




if __name__=='__main__':


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--channl_id')
    parser.add_argument('-g', '--gpu')


    args = parser.parse_args()
    



    chnl = int(args.channl_id)# int(input('channl_id = '))
    gpu = int(args.gpu) #int(input('gpu = '))

    train_unet_model(maebank_path='experiments/expb3',
                     channl_id=chnl, 
                     gpu_mae=gpu,
                     gpu_unet=gpu,
                     batch_size=256,
                     n_epochs=200,
                     lr=1e-3)






