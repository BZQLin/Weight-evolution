# Weight-evolution
This is the PyTorch implementation of our ACMM 2021 paper "Weight Evolution: Improving Deep Neural Networks Training through Evolving Inferior Weight Values"

In this paper, we throw up two important questions about network reactivation: how to identify and update bad genes？We then explained in detail the four strategies for solving the problems. 



## Prerequisites
Python 3.6+
Pytorch 1.2.0

## Execute example
### CIFAR
~~~  
CUDA_VISIBLE_DEVICES=0 nohup python we.py --cifar 10  --model mobilenet > WE/WE_cifar/sase/we_mobilenetv1_cifar10.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python we.py --cifar 10  --model resnet110 > WE/WE_cifar/sase/we_resnet110_cifar10.log 2>&1 &
~~~
### ImageNet
~~~  
CUDA_VISIBLE_DEVICES=0,1 nohup python we.py  --mov_model mobilenetv2 --scaling 0.25 --input-size 128 > WE/WE_imagenet/sase/we_mobilenetv2_128-128.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 nohup python we.py  --arch resnet34 --radio_conv 0.02 --radio_bn 0.02 > WE/WE_imagenet/sase/we_resnet34.log 2>&1 &
~~~

# References
For CIFAR, our code is based on https://github.com/kuangliu/pytorch-cifar.git

For ImageNet, our code is based on https://github.com/pytorch/examples/tree/master/imagenet

For MobileNetV2, our code is based on https://github.com/Randl/MobileNetV2-pytorch
