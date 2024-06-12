import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from utils import *
from parameters import *
import time

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)), flush=True)
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()), flush=True)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)), flush=True)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def compute_stats(G,P):
    """
    Compute statistics for the model performance
    Returns a dictionary
    """
    stats = {
        'rmse': np.sqrt(mse(G,P)),
        'mse': mse(G,P),
        'mae': mae(G,P),
        'pearson': pearson(G,P),
        'spearman': spearman(G,P),
        'ci': ci(G,P)
    }
    return stats


start = time.perf_counter()

datasets = [['davis', 'kiba', 'amyloid'][int(sys.argv[1])]]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

PARAMETERS = {'model': model_st, 'batch_size': TRAIN_BATCH_SIZE, 'lr': LR, 'epochs': NUM_EPOCHS}

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset, flush=True)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print('Device: ', device, 'cuda name:', cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        main_error_metric = 'mae' if dataset == 'amyloid' else 'mse'

        best_G = None
        best_P = None

        best_err = 1000
        best_epoch = -1
        model_file_name = 'training_results/model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'training_results/result_' + model_st + '_' + dataset +  '.csv'

        for epoch in range(NUM_EPOCHS):
            print(f"\nTraining epoch {epoch+1}... Start time: {time.asctime()}")

            train(model, device, train_loader, optimizer, epoch+1)
            G,P = predicting(model, device, test_loader)
            current_err = mse(G,P) if main_error_metric == 'mse' else mae(G,P)

            if current_err < best_err:
                best_G, best_P = G,P
                torch.save(model.state_dict(), model_file_name)
                best_epoch = epoch+1
                best_err = current_err
                print(f'{main_error_metric} improved at epoch {best_epoch}; best_{main_error_metric}: {best_err}', flush=True)
            else:
                print(f'No improvement since epoch {best_epoch}; best_{main_error_metric}: {best_err}', flush=True)

            seconds = time.perf_counter()-start
            print(f'Time passed: {int(seconds//3600)} hours, {int((seconds//60)%60)} mins, {int(seconds%60)} secs', flush=True)
        print("\n----------------------------------------------")
        print(f"Training completed!", flush=True)
        training_time_str = f'{int(seconds//3600)} hours {int((seconds//60)%60)} mins {int(seconds%60)} secs'

        # Compute statistics
        print(f"\nComputing statistics...", flush=True)
        ret = compute_stats(best_G, best_P)
        stats_df = pd.DataFrame([ret])
        print(stats_df.to_string(index=False), flush=True)
        stats_df.to_csv(result_file_name, index=False)

        # Display parameters
        print(f'\ndataset: {dataset}')
        for param, val in PARAMETERS.items():
            print(param + ':', val, flush=True)
        print(f'\nBest epoch: {best_epoch}\n', flush=True)

        # record the result in summary.csv
        with open('training_results/summary.csv', 'a') as f:
            stats_str = ','.join([str(val) for val in ret.values()])
            params_str = ','.join([str(val) for val in PARAMETERS.values()])

            f.write(f"{time.asctime()},{dataset},{params_str},{stats_str},{training_time_str}\n")
