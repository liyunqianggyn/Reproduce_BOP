import torch.nn as nn
import torch.nn.functional as F
from . import binaryfunction
import torch
import math
import torch.autograd as autograd

import numpy as np

from args import args as parser_args

    

class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()


        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        gain = nn.init.calculate_gain(parser_args.nonlinearity)
        std = gain / math.sqrt(fan)
        self.weight.data = (self.weight.data - self.weight.data.median()).sign() * std     

        
        
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        
    def forward(self, x):

        w = self.weight
            
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x        
