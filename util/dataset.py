import torch
import numpy as np
import yaml
from .process_data import NakanishiHandler

def trainSubjectIndependent(train_X,train_Y,subject,num_subs,
                            trials,classes,args,
                            split = 0.8, shuffle = True):
    
    # Takes non-target subjects and target sub
    subs_list = [i for i in range(num_subs)] #TODO: Still to check for Benchmark
    subs_list.remove(subject-1)
    
    train_x = train_X[subs_list,...]
    train_x = train_x.reshape((-1,train_x.shape[-2],train_x.shape[-1]))
    test_x = train_X[subject-1,...]
    test_x = test_x.reshape((-1,test_x.shape[-2],test_x.shape[-1]))
    train_y = train_Y[trials*classes:] 
    test_y  = train_Y[:trials*classes]

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

def testDataIndepedent(train_X):
    train_x = train_X.reshape((-1,train_X.shape[-2],train_X.shape[-1]))
    return train_x

if __name__ == '__main__':
    params_path = '/home/marcelo/Documentos/UTEC/Tesis I/FuzzySSVEPformer/datasets/Nakanishi.yaml'
    data_path = '/home/marcelo/Documentos/UTEC/Tesis I/FuzzySSVEPformer/datasets/2015_Nakanishi_SSVEP_database'
    with open(params_path) as f:
        params = yaml.load(f, Loader=yaml.loader.SafeLoader)
    datahandler = NakanishiHandler(params,1.0,data_path)
    train_x,train_y = datahandler.getAllSubjectsData()
    train_x = testDataIndepedent(train_x)

    import matplotlib.pyplot as plt
    plt.plot(train_x[0,0,:])
    plt.show()

