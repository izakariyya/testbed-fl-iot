# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:04:26 2022

@author: Usert990
"""


import numpy as np
#import time
#import memory_profiler
import torch
import torch.nn as nn
#import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from torchgpipe import GPipe
#from torchvision import datasets, transforms
import logging
import argparse
import sys

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.virtual import VirtualWorker
import fed_utils


logger = logging.getLogger(__name__)

LOG_INTERVAL = 25


#Get data set

class IoTDataset(Dataset):
    # Initialize your data, download, etc.

    def __init__(self):
        
        benign = np.loadtxt("benign_traffic.csv", delimiter = ",", dtype=np.float32)
        gafgyt = np.loadtxt("gafgyt_traffic.csv", delimiter = ",", dtype=np.float32)
        alldata = np.concatenate((benign, gafgyt))
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

torch.manual_seed(0)
# Define network dimensions
n_input_dim = 115
# Layer size
n_hidden1 = 128
n_hidden2 = 128 # Number of hidden nodes
n_output = 1
#eps = 0.001
lambd = 1e-4
#lambi = 0.01
cross_e = nn.BCELoss()

#Build and initialize network (model)
def nmodel():
    model = nn.Sequential(
            nn.Linear(n_input_dim, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_output),
            nn.Sigmoid()) 
    return model

# Cross Entropy Cost Function

#loss_function = nn.CrossEntropyLoss()

#def cross_entropy(input, target, eps):
#    input = torch.clamp(input,min=1e-7,max=1-1e-7)
#    bce = - (target * torch.log(input + eps) + (1 - target + eps) * torch.log(1 - input))
#    return torch.mean(bce)

# Regularized Cost

#def cross_reg(input, target, eps, lambd):
#    input = torch.clamp(input,min=1e-7,max=1-1e-7)
#    bce = - (target * torch.log(input + eps) + (1 - target + eps) * torch.log(1 - input))
#    rloss = (torch.mean(bce)) * lambd
#    return rloss

def train_on_batches(worker, batches, model_in, device, lr):
    """Train the model on the worker on the provided batches
    Args:
        worker(syft.workers.BaseWorker): worker on which the
        training will be executed
        batches: batches of data of this worker
        model_in: machine learning model, training will be done on a copy
        device (torch.device): where to run the training
        lr: learning rate of the training steps
    Returns:
        model, loss: obtained model and loss after training
    """
    model = model_in.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # TODO momentum is not supported at the moment

    model.train()
    model.send(worker)
    loss_local = False

    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = cross_e(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            logger.debug(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )

    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    model.get()  # <-- NEW: get the model back
    return model, loss

def train_on_e_batches(worker, batches, model_in, device, lr):
    """Train the model on the worker on the provided batches
    Args:
        worker(syft.workers.BaseWorker): worker on which the
        training will be executed
        batches: batches of data of this worker
        model_in: machine learning model, training will be done on a copy
        device (torch.device): where to run the training
        lr: learning rate of the training steps
    Returns:
        model, loss: obtained model and loss after training
    """
    model_n = nn.DataParallel(model_in)
    
    model = model_n.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay= lambd)  # TODO momentum is not supported at the moment

    model.train()
    model.send(worker)
    loss_local = False
    #grad_penalty = 0.1
    
    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)
        for param in model.parameters():
            param.grad = None
        #optimizer.zero_grad()   
        output = model(data)
        loss = cross_e(output, target)
            
        
        #for p in model.parameters():
        #    l2_norm = (p.pow(2.0) / pow(lambi, 2.0)) / (1 +p.pow(2.0) / pow(lambi, 2.0))
        #loss = loss + lambd * l2_norm
        
        #l1_parameters = []
        #for parameter in model.parameters():
        #    l1_parameters.append(parameter.view(-1))
        #l1 = torch.abs(torch.cat(l1_parameters)).sum()
        
        #loss += lambd * l1
        
        loss.backward()
        
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            logger.debug(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )

    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    model.get()  # <-- NEW: get the model back
    return model, loss

def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
    """retrieve next nr_batches of the federated data loader and group
    the batches by worker
    Args:
        fdataloader (sy.FederatedDataLoader): federated data loader
        over which the function will iterate
        nr_batches (int): number of batches (per worker) to retrieve
    Returns:
        Dict[syft.workers.BaseWorker, List[batches]]
    """
    batches = {}
    for worker_id in fdataloader.workers:
        worker = fdataloader.federated_dataset.datasets[worker_id].location
        batches[worker] = []
    try:
        for i in range(nr_batches): 
            next_batches = next(fdataloader)
            for worker in next_batches:
                batches[worker].append(next_batches[worker])
    except StopIteration:
        pass
    return batches

def train(model, device, federated_train_loader, lr, federate_after_n_batches):
    model.train()

    nr_batches = federate_after_n_batches

    models = {}
    loss_values = {}

    iter(federated_train_loader)  # initialize iterators
    batches = get_next_batches(federated_train_loader, nr_batches)
    counter = 0

    while True:
        logger.debug(
            "Starting training round, batches [{}, {}]".format(counter, counter + nr_batches)
        )
        data_for_all_workers = True
        for worker in batches:
            curr_batches = batches[worker]
            if curr_batches:
                models[worker], loss_values[worker] = train_on_batches(
                    worker, curr_batches, model, device, lr
                #models[worker], loss_values[worker] = train_on_e_batches(
                #   worker, curr_batches, model, device, lr
                )
            else:
                data_for_all_workers = False
        counter += nr_batches
        if not data_for_all_workers:
            logger.debug("At least one worker ran out of data, stopping.")
            break

        model = fed_utils.federated_avg(models)
        batches = get_next_batches(federated_train_loader, nr_batches)
    return model

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += cross_e(output, target).item()  # sum up batch loss
            #test_loss += cross_reg(output, target, eps, lambd).item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.debug("\n")
    accuracy = 100.0 * correct / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=1000, help="batch size used for the test data"
    )
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=50,
        help="number of training steps performed on each remote worker " "before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will " "be started in verbose mode",
    )
    parser.add_argument(
        "--use_virtual", action="store_true", help="if set, virtual workers will be used"
    )

    args = parser.parse_args(args=args)
    return args

def main():
    args = define_and_get_arguments()

    hook = sy.TorchHook(torch)

    if args.use_virtual:
        alice = VirtualWorker(id="alice", hook=hook, verbose=args.verbose)
        bob = VirtualWorker(id="bob", hook=hook, verbose=args.verbose)
        charlie = VirtualWorker(id="charlie", hook=hook, verbose=args.verbose)
        jane = VirtualWorker(id="jane", hook=hook, verbose=args.verbose)
    else:
        kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
        alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
        bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
        charlie = WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)
        jane = WebsocketClientWorker(id="jane", port=8780, **kwargs_websocket)

    workers = [alice, bob, charlie, jane]
    
    #workers = [alice, bob]

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    
    federated_train_loader = sy.FederatedDataLoader(
            trainset.federate(tuple(workers)),
            batch_size=args.batch_size,
            shuffle=True,
            iter_per_worker=True,
            **kwargs,
    )

    test_loader = DataLoader(
           dataset=testset,  batch_size=args.batch_size, shuffle=True, **kwargs,)
    
    model = nmodel().to(device)

    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, args.epochs)
        model = train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches)
        test(model, device, test_loader)
        #test(model, device, test_loader, eps, lambd)


    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main()