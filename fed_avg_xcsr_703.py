# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:38:58 2022

@author: -
"""

import numpy as np
import sys
import time
import memory_profiler
import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
import torch
from torch.utils.data import Dataset, DataLoader
#from torchvision import datasets, transforms
#from syft.frameworks.torch.federated import utils

import run_websocket_client_xiot as rwc
import logging

args = rwc.define_and_get_arguments(args=[])
use_cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(args)

#eps = 0.01
#lambd = 0.0001
#lambi = 0.01

hook = sy.TorchHook(torch)

kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
charlie = WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)
jane = WebsocketClientWorker(id="jane", port=8780, **kwargs_websocket)

workers = [alice, bob, charlie, jane]
print(workers)

#run this box only if the the next box gives pipeline error
#Get data set

class IoTDataset(Dataset):
    # Initialize your data, download, etc.

    def __init__(self):
        
        benign = np.loadtxt("benign_traffic.csv", delimiter = ",")
        mirai = np.loadtxt("mirai_traffic.csv", delimiter = ",")
        #gafgyt = np.loadtxt("gafgyt_traffic.csv", delimiter = ",")
        #alldata = np.concatenate((benign, gafgyt))
        alldata = np.concatenate((benign, mirai))
        j = len(benign[0])
        data = alldata[:, 1:j] 
        benlabel = alldata[:, 0]
        bendata = (data - data.min()) / (data.max() - data.min())
        self.len = alldata.shape[0]
        self.x_data = torch.from_numpy(bendata)
        self.y_data = torch.from_numpy(benlabel)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

full_dataset = IoTDataset()

train_size = int(len(full_dataset)* 0.8)
test_size = len(full_dataset) - train_size

# split the dataset
trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
trainset = trainset.dataset
testset = testset.dataset


federated_train_loader = sy.FederatedDataLoader(
            trainset.federate(tuple(workers)),
            batch_size= args.batch_size,
            shuffle=True,
            iter_per_worker=True
    )

test_loader = DataLoader(
           dataset=testset,  batch_size=args.batch_size, shuffle=True)

        
model = rwc.nmodel().to(device)
print(model)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s")
handler.setFormatter(formatter)
logger.handlers = [handler]

for epoch in range(1, args.epochs + 1):
    print("Starting epoch {}/{}".format(epoch, args.epochs))
    #starttbase = time.time()
    #startmbase = memory_profiler.memory_usage()
    starttefi = time.time()
    startmefi = memory_profiler.memory_usage()
    model = rwc.train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches)
    #endtbase =time.time()
    #endmbase = memory_profiler.memory_usage()
    #traintime_base = endtbase - starttbase
    #train_memory_base = endmbase[0] - startmbase[0]
    #print("Training time base: {:2f} sec".format(traintime_base))
    #print("Training memory base: {:2f} mb".format(train_memory_base))
    #rwc.test(model, device, test_loader)
    endtefi = time.time()
    endmefi = memory_profiler.memory_usage()
    traintime_efi = endtefi - starttefi
    train_memory_efi = endmefi[0] - startmefi[0]
    print("Training time optimize: {:2f} sec".format(traintime_efi))
    print("Training memory optimize: {:2f} mb".format(train_memory_efi))
    rwc.test(model, device, test_loader)
    