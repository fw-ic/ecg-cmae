import os
import numpy as np
from torch.utils.data import Dataset
from .data_handling import parse_headers
from .data_processing import *
from scipy.io import loadmat
from scipy import signal
import pickle
import neurokit2 as nk

class ECGPretrainDataset(Dataset):
    def __init__(self, data_root, crop_len=2500, mode='1D', channels='all', chn_idx=1):
        
        self.data_sources = next(os.walk(data_root))[1]
        self.data_sources.remove('st_petersburg_incart')
        self.data_sources.remove('ptb')        # removing this data source, small data with different sampling rate
        self.data_groups = {}
        self.crop_len = crop_len


        self.qualitiy_categories = ['Barely acceptable','Excellent','Mixed_Excellent','Mixed_Barely acceptable']

        all_ecg_data = []
        self.ecg_data = []

        for data_src in self.data_sources:
            self.data_groups[data_src] = next(os.walk(os.path.join(data_root,data_src)))[1]
            for grp in self.data_groups[data_src]:
                all_ecg_data.extend([os.path.join(data_root,data_src,grp,fl.split('.')[0]) for fl in next(os.walk(os.path.join(data_root,data_src,grp)))[2] if '.mat' in fl])


        for ecg_dt in all_ecg_data:
            cnt = 0
            signal_qualities = pickle.load(open(ecg_dt+'.samples','rb'))
            for qua_cat in self.qualitiy_categories:
                cnt += len(signal_qualities[qua_cat])
                if(cnt>0):
                    self.ecg_data.append(ecg_dt)
                    break


        self.ecg_data.sort()

        self.mode = mode
        self.channels = channels
        self.chn_idx = chn_idx
        

    def __len__(self):
        return len(self.ecg_data)

    def get_len(self, idx):

        sig = loadmat(self.ecg_data[idx]+'.mat')['val'] * 1.0
        return sig.shape[1]


    def __getitem__(self, idx, chn_idx= None, st_time=-1,maxmin_norma=False):

        sig = loadmat(self.ecg_data[idx]+'.mat')['val'] * 1.0
        
        signal_qualities = pickle.load(open(self.ecg_data[idx]+'.samples','rb'))

        st_times = []
        for qua_cat in self.qualitiy_categories:
            st_times.extend(signal_qualities[qua_cat])
        
        st_time = np.random.choice(st_times)
        #hdr = parse_headers(self.ecg_data[idx]+'.hea')
        
        if self.channels=='single':
            if chn_idx is None:
                chn_idx = self.chn_idx

                
            if chn_idx== -1:          # random
                sig = select_channel(sig, 'rand')        
            elif chn_idx== 1:
                sig = select_channel(sig, 'I')
            elif chn_idx== 2:
                sig = select_channel(sig, 'II')
            elif chn_idx== 3:
                sig = select_channel(sig, 'III')
            elif chn_idx== 4:
                sig = select_channel(sig, 'aVR')
            elif chn_idx== 5:
                sig = select_channel(sig, 'aVL')
            elif chn_idx== 6:
                sig = select_channel(sig, 'aVF')
            elif chn_idx== 7:
                sig = select_channel(sig, 'V1')
            elif chn_idx== 8:
                sig = select_channel(sig, 'V2')
            elif chn_idx== 9:
                sig = select_channel(sig, 'V3')
            elif chn_idx== 10:
                sig = select_channel(sig, 'V4')
            elif chn_idx== 11:
                sig = select_channel(sig, 'V5')
            elif chn_idx== 12:
                sig = select_channel(sig, 'V6')
            
            sig = np.expand_dims(sig, axis=0)

        #sig = temporal_crop(sig, 0, 5)
        sig, _ = samples_crop(sig, st_time, self.crop_len)      # -1 for random start position

        sig2 = []

        for i in range(12):
            sig2.append(nk.ecg_clean(sig[i], sampling_rate=500, method="neurokit"))

        sig = np.array(sig2)

        if(self.mode=='2D'):
                raise Exception("Sorry, currently should not work")
                f, t, Zxx = signal.stft(sig, 500, nperseg=448,noverlap=437)
                Zxx = np.log10(np.abs(Zxx)+1e-6)
                sig = Zxx[:224,:224]
            
        else:
            sig = (sig-np.mean(sig,axis=1,keepdims=True))/(np.std(sig,axis=1,keepdims=True)+1e-6)

            if(np.abs(np.std(sig)<1e-6)):
                sig = sig * 0
            else:
                if(maxmin_norma):
                    sig = (sig-np.min(sig))/(np.max(sig)-np.min(sig))        
        
        
        return sig


class ECGPretrainDatasetOld(Dataset):
    def __init__(self, data_root, crop_len=2500, mode='1D', channels='all', chn_idx=1):
        
        self.data_sources = next(os.walk(data_root))[1]
        self.data_sources.remove('st_petersburg_incart')
        self.data_sources.remove('ptb')        # removing this data source, small data with different sampling rate
        self.data_groups = {}
        self.crop_len = crop_len

        self.ecg_data = []

        for data_src in self.data_sources:
            self.data_groups[data_src] = next(os.walk(os.path.join(data_root,data_src)))[1]
            for grp in self.data_groups[data_src]:
                self.ecg_data.extend([os.path.join(data_root,data_src,grp,fl.split('.')[0]) for fl in next(os.walk(os.path.join(data_root,data_src,grp)))[2] if '.mat' in fl])

        self.ecg_data.sort()

        self.mode = mode
        self.channels = channels
        self.chn_idx = chn_idx
        

    def __len__(self):
        return len(self.ecg_data)

    def get_len(self, idx):

        sig = loadmat(self.ecg_data[idx]+'.mat')['val'] * 1.0
        return sig.shape[1]


    def __getitem__(self, idx, chn_idx= None, st_time=-1,maxmin_norma=False):

        sig = loadmat(self.ecg_data[idx]+'.mat')['val'] * 1.0
        #hdr = parse_headers(self.ecg_data[idx]+'.hea')
        
        if self.channels=='single':
            if chn_idx is None:
                chn_idx = self.chn_idx

                
            if chn_idx== -1:          # random
                sig = select_channel(sig, 'rand')        
            elif chn_idx== 1:
                sig = select_channel(sig, 'I')
            elif chn_idx== 2:
                sig = select_channel(sig, 'II')
            elif chn_idx== 3:
                sig = select_channel(sig, 'III')
            elif chn_idx== 4:
                sig = select_channel(sig, 'aVR')
            elif chn_idx== 5:
                sig = select_channel(sig, 'aVL')
            elif chn_idx== 6:
                sig = select_channel(sig, 'aVF')
            elif chn_idx== 7:
                sig = select_channel(sig, 'V1')
            elif chn_idx== 8:
                sig = select_channel(sig, 'V2')
            elif chn_idx== 9:
                sig = select_channel(sig, 'V3')
            elif chn_idx== 10:
                sig = select_channel(sig, 'V4')
            elif chn_idx== 11:
                sig = select_channel(sig, 'V5')
            elif chn_idx== 12:
                sig = select_channel(sig, 'V6')
            
            sig = np.expand_dims(sig, axis=0)

        #sig = temporal_crop(sig, 0, 5)
        sig, st_time2 = samples_crop(sig, st_time, self.crop_len)      # -1 for random start position



        if(self.mode=='2D'):
                raise Exception("Sorry, currently should not work")
                f, t, Zxx = signal.stft(sig, 500, nperseg=448,noverlap=437)
                Zxx = np.log10(np.abs(Zxx)+1e-6)
                sig = Zxx[:224,:224]
            
        else:
            sig = (sig-np.mean(sig,axis=1,keepdims=True))/(np.std(sig,axis=1,keepdims=True)+1e-6)

            if(np.abs(np.std(sig)<1e-6)):
                sig = sig * 0
            else:
                if(maxmin_norma):
                    sig = (sig-np.min(sig))/(np.max(sig)-np.min(sig))        
        

        return sig

