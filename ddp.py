import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
device = "cpu"
torch.set_num_threads(4)
import argparse
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

torch.manual_seed(0)

batch_size = 64 # batch for one node
is_distributed = True

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    print("TRAINING!")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 39 == 0:
            print('Training Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, batch_idx * len(data), len(train_loader), 100. * batch_idx / len(train_loader), loss.item()))
        # break

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main(rank, world_size):
    print("running rank {}".format(rank))

    normalize = transforms.Normalize(
        mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
        std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    )

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    training_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    training_sampler = DistributedSampler(training_set, rank=rank, num_replicas=world_size) if is_distributed else None
    train_loader = torch.utils.data.DataLoader(
        training_set,
        num_workers=world_size,
        batch_size=batch_size,
        sampler=training_sampler,
        shuffle=(training_sampler is None),
        pin_memory=True
    
    )

    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        num_workers=world_size,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    not_ddp = mdl.VGG11()
    model = DDP(not_ddp)
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    # running training for one epoch
    for epoch in range(1):
        training_sampler.set_epoch(epoch)
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-ip', type=str)
    parser.add_argument('--num-nodes', type=int)
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()

    dist.init_process_group(
        backend='gloo',
        init_method=args.master_ip,
        rank=args.rank,
        world_size=args.num_nodes
    )

    main(args.rank, args.num_nodes)
