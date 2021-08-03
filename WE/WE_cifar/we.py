from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
import argparse
import numpy as np
import time
import random
from models.resnet import *
from models.vgg import *
from models.mobilenetv2 import MobileNetV2
from models.mobilenet import MobileNet
from models.shufflenetv1 import shufflenet
import copy
# from models.densenet40 import *


parser = argparse.ArgumentParser(description='PyTorch Grafting Training')
# basic setting
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int) #gamma
parser.add_argument('--r', default=None, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--s', default='WE/base', type=str)
parser.add_argument('--model', default='mobilenet', type=str)
parser.add_argument('--cifar', default=10, type=int)
parser.add_argument('--print_frequence', default=1000, type=int)
parser.add_argument('--seed', default=2, type=int)
parser.add_argument('--alp', default=0, type=float)
parser.add_argument('--radio_conv', default=0.2, type=float)
parser.add_argument('--radio_bn', default=0.05, type=float)
parser.add_argument('--radio_bias', default=0.05, type=float)
parser.add_argument('--lab', default=0.1, type=float)
parser.add_argument('--ln_c', default=2, type=int)
args = parser.parse_args()
args.device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
print(args)
print('Session:%s\tModel:\tPID:%d' % (args.s, os.getpid()))
print(args.device)
# seed
def seed_torch(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

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
    net = shufflenet(args.cifar).to(args.device)
elif args.model == 'vgg16':
    net = vgg16(args.cifar).to(args.device)
elif args.model == 'vgg11':
    net = vgg11(args.cifar).to(args.device)
elif args.model == 'vgg19':
    net = vgg19(args.cifar).to(args.device)
elif args.model == 'vgg19_n':
    net = vgg19_n(args.cifar).to(args.device)
# elif args.model == 'densenet40':
#     net = DenseNet40(args.cifar).to(args.device)

print(time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime()))
start_epoch = 0
best_acc = 0
print('The initial learning rate is:', args.lr)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1) # 需要修改

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
        torch.save(state, '%sbest_%s.t7' % (args.s, args.model))
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

# sigmoid function
def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))

def valid_num(net, epoch):
    norm = torch.cat(
        [torch.norm(m1.weight.data.reshape(m1.weight.data.shape[0], -1), p=1, dim=1) / m1.weight.data.shape[1] for m1 in net.modules() if isinstance(m1, nn.Conv2d)])
    norm_bn = torch.cat(
        [m1.weight.data.abs() for m1 in net.modules() if isinstance(m1, nn.BatchNorm2d)])
    norm_bias = torch.cat(
        [m1.bias.data.abs() for m1 in net.modules() if isinstance(m1, nn.BatchNorm2d)])

    # 序号为auto-e3r5  后面用非线性来指导
    if epoch < 60:
        radio_conv = args.radio_conv * sigmoid(epoch/15)
        radio_bn = args.radio_bn * sigmoid(epoch/15)
        radio_bias = args.radio_bias * sigmoid(epoch/15)
    elif epoch < 120:
        radio_conv = (args.radio_conv / 2.5) * sigmoid((epoch-60)/15)
        radio_bn = (args.radio_bn / 2.5) * sigmoid((epoch-60)/15)
        radio_bias = (args.radio_bias / 2.5) * sigmoid((epoch-60)/15)
    else:
        radio_conv = (args.radio_conv / 5) * sigmoid((epoch-120)/15)
        radio_bn = (args.radio_bn / 5) * sigmoid((epoch-120)/15)
        radio_bias = (args.radio_bias / 5) * sigmoid((epoch-120)/15)

    div_conv = len(norm)*radio_conv
    div_bn = len(norm_bn)*radio_bn
    div_bias = len(norm_bias)*radio_bias
    global th_conv
    global th_bn
    global th_bias
    # global ln
    th_conv = sorted(norm)[int(div_conv)]
    th_bn = sorted(norm_bn)[int(div_bn)]
    th_bias = sorted(norm_bias)[int(div_bias)]
    print('Total filters number:\t', len(norm))
    print('invalid filters number of conv 0.1 :\t', int(sum((norm < 0.1).double())))
    print('invalid filters number of conv 0.1 :\t', int(sum((norm < 0.01).double())))
    print('ratio of conv:\t', int(sum((norm < th_conv).double())) / len(norm))
    print('Total filters number of bn:\t', len(norm_bn))
    print('invalid filters number of bn:\t', int(sum((norm_bn < th_bn).double())))
    print('ratio of bn:\t', int(sum((norm_bn < th_bn).double())) / len(norm_bn))
    print('radio_conv:%f, radio_bn:%f' % (radio_conv, radio_bn))
    return int(len(norm)), int(sum((norm < th_conv).double()))
    
def evolution(net):
    for m1 in net.modules():
        if isinstance(m1, nn.Conv2d):
            with torch.no_grad():
                num_filters = len(m1.weight.data)
                m1_norm = torch.norm(m1.weight.data.reshape(m1.weight.data.shape[0], -1), p=1, dim=1) / m1.weight.data.shape[1]
                low_fliter_index_per_m1, weight_sort_index = m1_norm.sort(descending=True)
                # # 最大核的展开
                # max_ori_squeeze = torch.squeeze(m1.weight.data[weight_sort_index[0]].reshape(-1,1)) # 展开
                # max_ori_squeeze_sort = sorted(max_ori_squeeze, key=abs,reverse=True)
                # # 最大核的最大值
                # max_ori = max_ori_squeeze_sort[0]
                hight_fliter_count_m1 = 0
                for i in low_fliter_index_per_m1:
                    # 大于  所以这里的'm1_count_cn = 会偏大一点 因为计算的是大于的那部分
                    if i >= th_conv:
                        hight_fliter_count_m1 += 1
                hight_fliter_count_m1_real = 0
                if hight_fliter_count_m1 != num_filters:
                    low_cut = [i for i in low_fliter_index_per_m1 if i.abs() <= (low_fliter_index_per_m1[0].abs())*args.lab]

                    hight_fliter_count_m1_real = len(low_cut)
                    global low_cut_count
                    if hight_fliter_count_m1_real == 0:
                        low_cut_count +=1
                        if low_cut_count >= ln_per:
                            print('cnslow_cut_count >= %d' % ln_per)
                            hight_fliter_count_m1_real = 1
                            low_cut_count = 0
                    else:
                        low_cut_count = 0
                # part1
                if hight_fliter_count_m1_real > (num_filters-hight_fliter_count_m1):
                    hight_fliter_count_m1_real = (num_filters-hight_fliter_count_m1)

                hight_fliter_count_m1_real_cut = num_filters - hight_fliter_count_m1_real

                weight_sort_index_index = weight_sort_index[hight_fliter_count_m1_real_cut:]  # 后面部分要替换掉的卷积核的下标
                weight_sort_index_index_hight = weight_sort_index[:hight_fliter_count_m1_real_cut]  # 后面部分要替换掉的卷积核的下标
                weight_sort_index_zip = [list(t) for t in zip(weight_sort_index_index_hight, weight_sort_index_index)]

                # 激活阶段： alp 
                for [i,j] in weight_sort_index_zip:
                    a_value1 = torch.norm(m1.weight.data[j].reshape(1, -1), p=1, dim=1)
                    a_value2 = torch.norm(m1.weight.data[i].reshape(1, -1), p=1, dim=1)
                    alp = a_value1.abs() / (a_value1.abs() + a_value2.abs())
                    if m1.weight.data[j].shape[1] == 1:
                        m1.weight.data[j] = m1.weight.data[j]*alp + m1.weight.data[i]*(1-alp)
                    else:
                        filter_piece_zip = [list(t) for t in zip(m1.weight.data[i], m1.weight.data[j])]
                        for h,[p,q] in enumerate(filter_piece_zip):
                            large_one_squ = torch.squeeze(p.reshape(1, -1))
                            small_one_squ = torch.squeeze(q.reshape(1, -1))
                            large_one_sort = sorted(large_one_squ, key=abs,reverse=True)
                            small_one_sort = sorted(small_one_squ, key=abs,reverse=False)  # 可以换一下
                            alp = small_one_sort[0] / (small_one_sort[0] + large_one_sort[0])  # 元素
                            weight_ele = large_one_sort[0] * alp + small_one_sort[0] * (1-alp)
                            weight_ele = torch.squeeze(weight_ele)
                            large_one_tensor_fill = torch.ones_like(p).fill_(weight_ele)
                            q = torch.where(small_one_sort[0]==q, large_one_tensor_fill, q)
                            m1.weight.data[j][h] = q

        if isinstance(m1, nn.BatchNorm2d):
            with torch.no_grad():
                num_filters = len(m1.weight.data)
                low_fliter_index_per_m1, weight_sort_index_m1 = m1.weight.data.abs().sort(descending=True) 
                # max_ori = m1.weight.data[weight_sort_index_m1[0]]
                low_fliter_index_per_b1, weight_sort_index_b1 = m1.bias.data.abs().sort(descending=True) # 小到大
                # max_ori2 = m1.bias.data[weight_sort_index_b1[0]]
                hight_fliter_count_m1 = 0
                for i in low_fliter_index_per_m1:
                    if i >= th_bn:
                        hight_fliter_count_m1 += 1
                hight_fliter_count_m1_real = 0
                if hight_fliter_count_m1 != num_filters:
                    low_cut_bn = [i for i in low_fliter_index_per_m1 if i.abs() <= (low_fliter_index_per_m1[0].abs())*args.lab]
                    hight_fliter_count_m1_real = len(low_cut_bn)
                    global low_cut_count_bn
                    if hight_fliter_count_m1_real == 0:
                        low_cut_count_bn +=1
                        if low_cut_count_bn >= ln_per:
                            print('bnslow_cut_count >= %d'%ln_per)
                            hight_fliter_count_m1_real = 1
                            low_cut_count_bn = 0
                    else:
                        low_cut_count_bn = 0
                # part
                if hight_fliter_count_m1_real > (num_filters-hight_fliter_count_m1):
                    hight_fliter_count_m1_real = (num_filters-hight_fliter_count_m1)
                hight_fliter_count_m1_real_cut = num_filters - hight_fliter_count_m1_real

                hight_fliter_count_b1 = 0
                for i in low_fliter_index_per_b1:
                    if i >= th_bias:
                        hight_fliter_count_b1 += 1
                hight_fliter_count_b1_real = 0
                if hight_fliter_count_b1 != num_filters:
                    low_cut_bias = [i for i in low_fliter_index_per_b1 if i.abs() <= (low_fliter_index_per_b1[0].abs())*args.lab]
                    hight_fliter_count_b1_real = len(low_cut_bias)
                    global low_cut_count_bias
                    if hight_fliter_count_b1_real == 0:
                        low_cut_count_bias +=1
                        if low_cut_count_bias >= ln_per:
                            hight_fliter_count_b1_real = 1
                            low_cut_count_bias = 0
                    else:
                        low_cut_count_bias = 0
                # part
                if hight_fliter_count_b1_real > (num_filters-hight_fliter_count_b1):
                    hight_fliter_count_b1_real = (num_filters-hight_fliter_count_b1)
                hight_fliter_count_b1_real_cut = num_filters - hight_fliter_count_b1_real

                weight_sort_index_re2_1 = weight_sort_index_m1[hight_fliter_count_m1_real_cut:]
                weight_sort_index_re2_1_hight = weight_sort_index_m1[:hight_fliter_count_m1_real_cut]
                weight_sort_index_re2_2 = weight_sort_index_b1[hight_fliter_count_b1_real_cut:]
                weight_sort_index_re2_2_hight = weight_sort_index_b1[:hight_fliter_count_b1_real_cut]
                weight_sort_index_m1_zip = [list(t) for t in zip(weight_sort_index_re2_1_hight, weight_sort_index_re2_1)]
                weight_sort_index_b1_zip = [list(t) for t in zip(weight_sort_index_re2_2_hight, weight_sort_index_re2_2)]
            
                for [i,j] in weight_sort_index_m1_zip:
                    alp = m1.weight.data[j].abs() / (m1.weight.data[j].abs() + m1.weight.data[i].abs())
                    m1.weight.data[j] = m1.weight.data[j] * alp + m1.weight.data[i] * (1-alp)
                for [i,j] in weight_sort_index_b1_zip:
                    alp = m1.bias.data[j].abs() / (m1.bias.data[j].abs() + m1.bias.data[i].abs())
                    m1.bias.data[j] = m1.bias.data[j] * alp + m1.bias.data[i] * (1-alp)
    return net
if __name__ == '__main__':
    low_cut_count = 0
    low_cut_count_bn = 0
    low_cut_count_bias = 0

    th_conv = 0 
    th_bn = 0
    th_bias = 0
    ln = 0
    for m1 in net.modules():
        if isinstance(m1, nn.Conv2d):
            ln += 1
    print('ln:', ln)
    ln_per = ln / args.ln_c 
    
    for epoch in range(start_epoch, args.epochs):
        # 训练和测试的部分 
        print(time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime()))
        start_time = time.time()
        train(epoch)
        print('train time: %s', time.time() - start_time)
        test(epoch)
        total_filter_conv, total_filter_bn = valid_num(net,epoch)
        net = evolution(net)
