import torch
import numpy as np
import yaml
from scipy import signal
from .process_data import NakanishiHandler

def trainSubjectIndependent(train_X,train_Y,subject,num_subs,
                            trials,classes,args,
                            split = 0.8, shuffle = True, fs = 250, fb = False, bands = 'all'):
    
    # Takes non-target subjects and target sub
    subs_list = [i for i in range(num_subs)] #TODO: Still to check for Benchmark
    subs_list.remove(subject-1)
    
    train_x = train_X[subs_list,...]
    train_x = train_x.reshape((-1,train_x.shape[-2],train_x.shape[-1]))
    test_x = train_X[subject-1,...]
    test_x = test_x.reshape((-1,test_x.shape[-2],test_x.shape[-1]))
    train_y = train_Y[trials*classes:] 
    test_y  = train_Y[:trials*classes]

    print(bands)
    if fb:
        fb_train_x = np.swapaxes(filterbank(fs,train_x),0,1)
        fb_test_x = np.swapaxes(filterbank(fs,test_x),0,1)
        if not(bands == "all"):
            train_x = fb_train_x[:,bands,...]
            test_x = fb_test_x[:,bands,...]
        else:
            train_x = fb_train_x
            test_x = fb_test_x

    # Dataset creation from numpy file to Torch class
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y).type(torch.int64)
    trainset   = torch.utils.data.TensorDataset(train_x, train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y).type(torch.int64)
    testset   = torch.utils.data.TensorDataset(test_x, test_y)
    
    # it includes train-val-test split under sub.independent scheme 
    train_size = int(split * len(trainset))
    valid_size = len(trainset) - train_size
    trainset, validset = torch.utils.data.random_split(trainset, [train_size, valid_size])
    
    #
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=shuffle,
                                               num_workers=args.num_workers,pin_memory=args.pin_mem)
    val_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size,shuffle=shuffle,
                                             num_workers=args.num_workers,pin_memory=args.pin_mem)
    test_loader= torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.num_workers,pin_memory=args.pin_mem)

    return train_loader, val_loader, test_loader


def filterbank(fs,X,num_subbands = 3):
    # https://github.com/pikipity/SSVEP-Analysis-Toolbox/blob/main/SSVEPAnalysisToolbox/utils/nakanishipreprocess.py
    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1], X.shape[2]))
    for k in range(1, num_subbands+1, 1):
        Wp = [(8*k)/(fs/2), 80/(fs/2)]
        Ws = [(8*k-2)/(fs/2), 90/(fs/2)]
        N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)

        bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')

        tmp = signal.filtfilt(bpB, bpA, X, axis = -1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
        filterbank_X[k-1,...] = tmp

    return filterbank_X
