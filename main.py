import argparse
import torch
import time 
import datetime
import yaml
import csv
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim

from models import *
from util import NakanishiHandler,BenchmarkHandler,UTECHandler,trainSubjectIndependent,seed_everything
from util.engine import EarlyStopping,train_one_epoch,evaluate

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EfficientFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='ssvepformer', type=str, choices=['ssvepformer', 
                                                                             'f1ssvepformer_A', 'f1ssvepformer_B','f1ssvepformer_C',
                                                                             'f2ssvepformer_A', 'f2ssvepformer_B','f2ssvepformer_C'],
                        help='Name of model to train')
    parser.add_argument('--signal_size', default=1,
                        type=float, help='signal sample size')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 1e-3)')

    # Dataset parameters
    parser.add_argument('--data_path', default='datasets/Tsinghua', type=str,
                        help='dataset path')
    parser.add_argument('--params_path', default='datasets/Benchmark.yaml', type=str,
                        help='Parameter dataset path')
    parser.add_argument('--dataset', default='BENCHMARK', choices=['NAKANISHI', 'BENCHMARK', 'UTEC'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):

    # Setup
    with open(args.params_path) as f:
        params = yaml.load(f, Loader=yaml.loader.SafeLoader)
    device = torch.device(args.device)
    seed_everything()
    cudnn.benchmark = True

    # Creata results file
    filename = os.path.join(args.output_dir,f'{args.model}_e{args.epochs}_{args.dataset}.csv')
    if not(os.path.exists(filename)):
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Time']+[str(i+1) for i in range(params['Subs'])])

    # Data handler choosing
    match args.dataset:
        case'NAKANISHI':
            datahandler = NakanishiHandler(params,args.signal_size,args.data_path)
        case 'BENCHMARK':
            datahandler = BenchmarkHandler(params,args.signal_size,args.data_path)
        case 'UTEC':
            datahandler = UTECHandler(params,args.signal_size,args.data_path)
    train_x,train_y = datahandler.getAllSubjectsData()

    # Checking signal length 
    if args.signal_size in params['Check_length'] and args.dataset == 'NAKANISHI':
        length = int(params['Fs']*args.signal_size*2) - 1   
    else:
        length = int(params['Fs']*args.signal_size*2)

    # Main training/validation and testing loop
    print(f"SSVEPformer: Training {args.model} net for {args.epochs} epochs/{args.batch_size} batch")
    test_accuracy = []
    for subject in range(params['Subs']):
        data_loader_train, data_loader_val, data_loader_test = trainSubjectIndependent(train_x,train_y,subject+1,
                                                                        params['Subs'],params['Trials'],params['Classes'],args)                           
        model_args = [params['Channels'],params['Classes'],length]
        match args.model:
            case 'ssvepformer':
                model = SSVEPformer(*model_args)
            case'f1ssvepformer_A':
                model = fuzzySSVEPformerA(*model_args)
            case 'f1ssvepformer_B':
                model = fuzzySSVEPformerB(*model_args)
            case'f1ssvepformer_C':
                model = fuzzySSVEPformerC(*model_args)
            case 'f2ssvepformer_A':
                model = fuzzyT2SSVEPformerA(*model_args)
            case 'f2ssvepformer_B':
                model = fuzzyT2SSVEPformerB(*model_args)
            case'f2ssvepformer_C':
                model = fuzzyT2SSVEPformerC(*model_args)

        model.apply(init_normal)
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        start_time = time.time()
        train_losses = np.zeros(args.epochs)
        valid_losses = np.zeros(args.epochs)
        print(f'Subject {subject + 1}')
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(data_loader_train,
                                        model, criterion, 
                                        optimizer, device)
            valid_loss,_ = evaluate(data_loader_val, model, criterion, device)
            train_losses[epoch] = train_loss/len(data_loader_train)
            valid_losses[epoch] = valid_loss/len(data_loader_val)
            print(f'Epoch {epoch}/{args.epochs} / train loss:{train_losses[epoch]:.4f} / val loss:{valid_losses[epoch]:.4f}')

        # Reports    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time))) 
        print('Training time {}'.format(total_time_str))

        _,test_acc = evaluate(data_loader_test, model, criterion, device)
        test_accuracy.append(round(test_acc,4))
        print('Test accuracy {:.3f}'.format(test_acc))
        print('')

        import matplotlib.pyplot as plt
        plt.plot(train_losses)
        plt.plot(valid_losses)
        plt.title(f'Train/Valid loss of Sub {subject}')
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Loss (Cross entropy)')
        plt.legend(['Train','Valid'])
        plt.show()

    # save results
    with open(filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([args.signal_size] + test_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSVEPformer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)