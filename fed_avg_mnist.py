# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 08:17:23 2021

@author: Usert990
"""


import sys
import time
import memory_profiler
import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
import torch
from torchvision import datasets, transforms

#from syft.frameworks.torch.federated import utils

import frun_websocket_client as rwc
import logging

args = rwc.define_and_get_arguments(args=[])
use_cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(args)

hook = sy.TorchHook(torch)

kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
charlie = WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)
jane = WebsocketClientWorker(id="jane", port=8780, **kwargs_websocket)

workers = [alice, bob, charlie, jane]
print(workers)


#run this box only if the the next box gives pipeline error
torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,download=True))

federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ).federate(tuple(workers)),
    batch_size=args.batch_size,
    shuffle=True,
    iter_per_worker=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True
)
        
model = rwc.Net().to(device)
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
    #endtbase = time.time()
    #endmbase = memory_profiler.memory_usage()
    #traintime_base =  endtbase - starttbase 
    #train_memory_base = endmbase[0] - startmbase[0] 
    #print("Training time base: {:2f} sec".format(traintime_base / 4.0))
    #print("Training memory base: {:2f} mb".format(train_memory_base / 4.0))
    #rwc.test(model, device, test_loader)
    endtefi = time.time()
    endmefi = memory_profiler.memory_usage()
    traintime_efi =  endtefi - starttefi
    train_memory_efi = endmefi[0] - startmefi[0]
    print("Training time optimize: {:2f} sec".format(traintime_efi / 4.0))
    print("Training memory optimize: {:2f} mb".format(train_memory_efi / 4.0))
    rwc.test(model, device, test_loader)