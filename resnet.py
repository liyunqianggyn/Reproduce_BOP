'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import ir_1w1a

from torch.autograd import Variable

__all__ = ['resnet20_1w1a', 'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        input[input ==0] = -1
        
# =============================================================================
#         grad_input = input.clone()
#         k = 0.8
#         s = grad_input.data.size()
#         out = grad_input.clone()
#         out = out.sum(0, keepdim=True)
#         _, idx = out.abs().flatten().sort()
#         j = int((k) * grad_input.numel())
# 
#         # flat_out and out access the same memory.
#         flat_out = out.flatten()
#         flat_out[idx[:j]] = 0
#         flat_out[idx[j:]] = 1
#         grad_input.mul_(out.expand(s))        
# =============================================================================
        
        
        return input
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        
        grad_input = grad_output.clone()
        
        
# =============================================================================
#         import matplotlib.pyplot as plt
#         font2 = { 'weight' : 'normal',
#         'size'   : 12,
#         }   
#         font1 = { 'weight' : 'normal',
#         'size'   : 14,
#         }              
#         plt.figure(figsize=(5.0, 4.0) )  
#         plt.hist(grad_input.flatten(), bins=100, color='r', label='Layer{}'.format(0 +1)) 
#         plt.legend(prop=font2, loc='best')
#         plt.xlabel(r'Gradient',font1)
#         plt.ylabel('Frequency',font1)
#         plt.tick_params(labelsize=13.5)  
#         plt.show()
#         print(grad_input.abs().max())                   
# =============================================================================
       # grad_input.mul_((grad_input.var() + 1e-9))
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        
        
        
# =============================================================================
#         # Get the subnetwork by sorting the scores and using the top k%
#         k = 0.8
#         s = grad_input.data.size()
#         out = grad_input.clone()
#         out = out.abs().sum(0, keepdim=True)
#         _, idx = out.abs().flatten().sort()
#         j = int((k) * grad_input.numel())
# 
#         # flat_out and out access the same memory.
#         flat_out = out.flatten()
#         flat_out[idx[:j]] = 0
#         flat_out[idx[j:]] = 1
#         grad_input.mul_(out.expand(s))      
# =============================================================================

        
        


# =============================================================================
#         import matplotlib.pyplot as plt
#         font2 = { 'weight' : 'normal',
#         'size'   : 12,
#         }   
#         font1 = { 'weight' : 'normal',
#         'size'   : 14,
#         }              
#         plt.figure(figsize=(5.0, 4.0) )  
#         plt.hist(grad_input.flatten(), bins=100, color='r', label='Layer{}'.format(0 +1)) 
#         plt.legend(prop=font2, loc='best')
#         plt.xlabel(r'Gradient',font1)
#         plt.ylabel('Frequency',font1)
#         plt.tick_params(labelsize=13.5)  
#         plt.show()
#         print(grad_input.abs().max())     
# =============================================================================

        return grad_input
def hash_layer(input):
    return BinActive.apply(input)    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = ir_1w1a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w1a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ir_1w1a.IRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = F.hardtanh(out)
        return out



class BasicBlock_1w1a_prelu(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1w1a_prelu, self).__init__()
        self.conv1 = ir_1w1a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = ir_1w1a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ir_1w1a.IRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):   
     
# =============================================================================
#         residual = x
#         out = BinActive()(x)
#         out = self.conv1(out)
#         out = self.prelu(out)
#         out = self.bn1(out)
#         
#         out += self.shortcut(residual)
#      
#         x1 = out
#         out = BinActive()(out)
#         out = self.conv2(out)
#         out = self.prelu(out)
#         out = self.bn2(out)        
#         out += x1
#         return out
#         
# =============================================================================
        
        residual = x
        out = hash_layer(x)
        out = self.bn1(self.conv1(out))
        out += self.shortcut(residual)
        out =  self.prelu(out) 
        x1 = out
        out = hash_layer(out)
        out = self.bn2(self.conv2(out))
        out += x1
        out =  self.prelu(out)
        return out

class BasicBlock_1w1a_relu(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1w1a_relu, self).__init__()
        self.conv1 = ir_1w1a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = ir_1w1a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ir_1w1a.IRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):
        residual = x
        out = x
        out = self.bn1(self.conv1(out))
        out += self.shortcut(residual)
        out =  self.prelu(out) 
        x1 = out
#        out = BinActive()(out)
        out = self.bn2(self.conv2(out))
        out += x1
        out =  self.prelu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        init.kaiming_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)
        init.kaiming_normal(self.linear.weight)

 #       self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out


# =============================================================================
# def resnet20_1w1a():
#     return ResNet(BasicBlock_1w1a, [3, 3, 3])
# =============================================================================

def resnet20_1w1a():
    return ResNet(BasicBlock_1w1a_prelu, [3, 3, 3])

# =============================================================================
# def resnet20_1w1a():
#     return ResNet(BasicBlock_1w1a_relu, [3, 3, 3])
# =============================================================================

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
