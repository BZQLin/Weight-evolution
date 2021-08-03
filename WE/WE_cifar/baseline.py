from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import collections
import time
import argparse
import sys
import pickle
import numpy as np
import time
from models.resnet import *
from models.vgg import *
from models.mobilenetv2 import MobileNetV2
from models.mobilenet import MobileNet
import random
# from models.densenet import *
from models.shufflenetv1 import shufflenet


parser = argparse.ArgumentParser(description='PyTorch Grafting Training')
# basic setting
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--r', default=None, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--s', default='WE/base', type=str)
parser.add_argument('--model', default='mobilenet', type=str)
parser.add_argument('--cifar', default=10, type=int)
parser.add_argument('--print_frequence', default=1000, type=int)
parser.add_argument('--seed', default=1029, type=int)
args = parser.parse_args()
# args.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(args)
print('PID:%d' % (os.getpid()))

# def seed_torch(seed=args.seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# seed_torch()

if args.model == 'resnet20':
    net = resnet20(args.cifar).to(args.device)
elif args.model == 'resnet32':
    net = resnet32(args.cifar).to(args.device)
elif args.model == 'resnet56':
    net = resnet56(args.cifar).to(args.device)
elif args.model == 'resnet110':
    net = resnet110(args.cifar).to(args.device)
elif args.model == 'mobilenetv2':
    net = MobileNetV2(args.cifar).to(args.device)
elif args.model == 'mobilenet':
    net = MobileNet(args.cifar).to(args.device)
elif args.model == 'shufflenetv1':
    net = shufflenet(num_classes=args.cifar).to(args.device)
elif args.model == 'vgg16':
    net = vgg16(args.cifar).to(args.device)
elif args.model == 'vgg11':
    net = vgg11(args.cifar).to(args.device)
elif args.model == 'vgg19':
    net = vgg19(args.cifar).to(args.device)
# elif args.model == 'densenet40':
#     net = DenseNet40_4(args.cifar).to(args.device)
elif args.model == 'vgg19_n':
    net = vgg19_n(args.cifar).to(args.device)
    
# net=torch.nn.DataParallel(net)
print(net)
print(time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime()))
start_epoch = 0
best_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.cifar == 10:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.cifar == 100:
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': net.state_dict(),
            'acc': acc
        }
        torch.save(state, '%s/best_%s.t7' % (args.s, args.model))
    print('epoch:%d    accuracy:%.3f    best:%.3f' % (epoch, acc, best_acc))


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % args.print_frequence == args.print_frequence - 1 or args.print_frequence == trainloader.__len__() - 1:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    lr_scheduler.step()


if __name__ == '__main__':
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train(epoch)
        test(epoch)
        data_time = time.time() - start_time
        print(time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime()))
        print('time: %s', data_time)
