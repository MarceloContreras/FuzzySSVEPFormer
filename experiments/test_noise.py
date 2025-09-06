import os
import torch
import yaml
import argparse
import numpy as np 
import torch.backends.cudnn as cudnn

from braindecode.models import EEGNetv4
from models import *
from util import NakanishiHandler,BenchmarkHandler,UTECHandler,WearableHandlerDry,WearableHandlerWet,trainSubjectIndependent,seed_everything
from util.engine import get_latent_space

def get_args_parser():
    parser = argparse.ArgumentParser(
        'SSVEPformer evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--model', default='ssvepformer', type=str, choices=['ssvepformer', 
                                                                             'f1ssvepformer_A', 'f1ssvepformer_B','f1ssvepformer_C',
                                                                             'f2ssvepformer_A', 'f2ssvepformer_B','f2ssvepformer_C',
                                                                             'EEGNet','SSVEPNet','Conformer','Deformer'],
                        help='Name of model to train')
    parser.add_argument('--signal_size', default=1.0,
                        type=float, help='signal sample size')
    parser.add_argument('--sub', default = 1, type = int)

    # Dataset parameters
    parser.add_argument('--data_path', default='datasets/2015_Nakanishi_SSVEP_database', type=str,
                        help='dataset path')
    parser.add_argument('--params_path', default='datasets/Nakanishi.yaml', type=str,
                        help='Parameter dataset path')
    parser.add_argument('--dataset', default='NAKANISHI', choices=['NAKANISHI', 'BENCHMARK', 'UTEC', 'WEARABLE_DRY', 'WEARABLE_WET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--test_noise', action='store_true', default = False,
                        help='')
    parser.add_argument('--test_time', action='store_true', default = False,
                        help='')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):
    # Setup
    with open(args.params_path) as f:
        params = yaml.load(f, Loader=yaml.loader.SafeLoader)

    device = torch.device(args.device)
    seed_everything()
    cudnn.benchmark = True

    # Build model and load weights
    match args.model:
        case 'ssvepformer':
            model = SSVEPformer(params,args.signal_size)
        case'f1ssvepformer_A':
            model = fuzzySSVEPformerA(params,args.signal_size)
        case 'f1ssvepformer_B':
            model = fuzzySSVEPformerB(params,args.signal_size)
        case'f1ssvepformer_C':
            model = fuzzySSVEPformerC(params,args.signal_size)
        case 'f2ssvepformer_A':
            model = fuzzyT2SSVEPformerA(params,args.signal_size)
        case 'f2ssvepformer_B':
            model = fuzzyT2SSVEPformerB(params,args.signal_size)
        case'f2ssvepformer_C':
            model = fuzzyT2SSVEPformerC(params,args.signal_size)
        case 'EEGNet':
            model = EEGNetv4(params['Channels'],params['Classes'],int(args.signal_size*params['Fs'])+25)
        case 'SSVEPNet':
            model = ESNet(params['Channels'],round(args.signal_size*params['Fs']),params['Classes'])
        case 'Conformer':
            model = Conformer(n_classes=params['Classes'],n_chan=params['Channels'],n_samples = round(args.signal_size*params['Fs']))
        case 'Deformer':
            model = Deformer(num_chan=params['Channels'], num_time=round(args.signal_size*params['Fs'])+25, temporal_kernel=11, num_kernel=64,
                            num_classes=params['Classes'], depth=4, heads=16,
                            mlp_dim=16, dim_head=16, dropout=0.5)

    weights_path = "weights"
    model_name = f"{args.model}_{args.signal_size:.1f}s_{args.dataset}_S{args.sub+1}.pth"
    save_path = os.path.join(weights_path, model_name)        
    
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    
    # Create folder to save embeddings
    if not os.path.exists("latent_space"):
        os.makedirs("latent_space")

    if args.test_time:
        for time_length in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            match args.dataset:
                case'NAKANISHI':
                    datahandler = NakanishiHandler(params,args.signal_size,args.data_path)
                case 'BENCHMARK':
                    datahandler = BenchmarkHandler(params,args.signal_size,args.data_path)
                case 'UTEC':
                    datahandler = UTECHandler(params,args.signal_size,args.data_path)
                case 'WEARABLE_WET':
                    datahandler = WearableHandlerWet(params,args.signal_size,args.data_path)
                case 'WEARABLE_DRY':
                    datahandler = WearableHandlerDry(params,args.signal_size,args.data_path)

            train_x,train_y = datahandler.getAllSubjectsData()
            _,_, data_loader_test = trainSubjectIndependent(train_x,train_y,args.sub,
                                                            params['Subs'],
                                                            params['Trials'],
                                                            params['Classes'],
                                                            device,
                                                            args,
                                                            0.8,
                                                            fs=params['Fs'])
            
            X_latent,Y_latent_space = get_latent_space(data_loader_test,model,device)
            np.save(f"latent_space/X_{args.model}_sub{args.sub}_t{time_length}",X_latent)
            np.save(f"latent_space/Y_{args.model}_sub{args.sub}_t{time_length}",Y_latent_space)

    if args.test_noise:
        for noise in [1e0,1e1,1e2,1e3,1e4,1e5]:
            match args.dataset:
                case'NAKANISHI':
                    datahandler = NakanishiHandler(params,args.signal_size,args.data_path)
                case 'BENCHMARK':
                    datahandler = BenchmarkHandler(params,args.signal_size,args.data_path)
                case 'UTEC':
                    datahandler = UTECHandler(params,args.signal_size,args.data_path)
                case 'WEARABLE_WET':
                    datahandler = WearableHandlerWet(params,args.signal_size,args.data_path)
                case 'WEARABLE_DRY':
                    datahandler = WearableHandlerDry(params,args.signal_size,args.data_path)

            train_x,train_y = datahandler.getAllSubjectsData()
            train_x += np.random.normal(0,noise,train_x.shape)
            _,_, data_loader_test = trainSubjectIndependent(train_x,train_y,args.sub,
                                                            params['Subs'],
                                                            params['Trials'],
                                                            params['Classes'],
                                                            device,
                                                            args,
                                                            0.8,
                                                            fs=params['Fs'])
            
            X_latent,Y_latent_space = get_latent_space(data_loader_test,model,device)
            np.save(f"latent_space/X_{args.model}_sub{args.sub}_t{args.signal_size}_n{noise}",X_latent)
            np.save(f"latent_space/Y_{args.model}_sub{args.sub}_t{args.signal_size}_n{noise}",Y_latent_space)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSVEPformer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)