import numpy as np


def select_channel(sigs, channel='rand'):

    if channel=='rand':
        return sigs[np.random.randint(12),:]

    if channel=='I':
        return sigs[0,:]
    elif channel=='II':
        return sigs[1,:]
    elif channel=='III':
        return sigs[2,:]
    elif channel=='aVR':
        return sigs[3,:]
    elif channel=='aVL':
        return sigs[4,:]
    elif channel=='aVF':
        return sigs[5,:]
    elif channel=='V1':
        return sigs[6,:]
    elif channel=='V2':
        return sigs[7,:]
    elif channel=='V3':
        return sigs[8,:]
    elif channel=='V4':
        return sigs[9,:]
    elif channel=='V5':
        return sigs[10,:]
    elif channel=='V6':
        return sigs[11,:]


#def temporal_crop(sig, st_time, duration, FS=500):
#
#    return sig[st_time*FS:(st_time+duration)*FS]


def samples_crop(sig, st_time, crop_len):

    sig_len = sig.shape[1]

    if(crop_len==sig_len):
        return sig,0
    if(crop_len>sig_len):
        return np.hstack([sig, np.zeros((sig.shape[0],crop_len-sig_len))]),0

    

    if st_time==-1:
        st_time = np.random.randint(0,sig_len-crop_len)
    
    #print(st_time,st_time+crop_len,len(sig))
    return sig[:,st_time:st_time+crop_len], st_time