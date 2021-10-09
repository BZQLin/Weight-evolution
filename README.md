# Weight-evolution
Weight Evolution: Improving Deep Neural Networks Training through Evolving Inferior Weight Values


## Prerequisites
Python 3.6+
Pytorch 1.2.0

## Execute example
### CIFAR
~~~  
CUDA_VISIBLE_DEVICES=0 nohup python we.py --cifar 10  --model mobilenet > WE/base/mobilenetv1_cifar10.log 2>&1 &
~~~
### ImageNet
~~~  
CUDA_VISIBLE_DEVICES=0,1 nohup python we.py  --mov_model mobilenetv2 --scaling 0.25 --input-size 128 > WE/base/mobilenetv2_128-128.log 2>&1 &
~~~

# References
For CIFAR, our code is based on https://github.com/kuangliu/pytorch-cifar.git

For ImageNet, our code is based on https://github.com/pytorch/examples/tree/master/imagenet

For Mobilenetv2, our code is based on https://github.com/Randl/MobileNetV2-pytorch
