import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.lcn = nn.LocalResponseNorm(40)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = F.relu(self.lcn(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)
        
        x = F.max_pool2d(self.conv2(x), kernel_size=2)
        
        return self.flatten(x)
    

class Model(nn.Module):
    
    def __init__(self, number_of_classes):
        super().__init__()
        
        self.block1 = ConvBlock()
        self.block2 = ConvBlock()
        
        self.linear = nn.Linear(6400*2, number_of_classes)
        
    def forward(self, x):
        
        block1_output = self.block1(x)
        block2_output = self.block2(x)
        
        concatenated = torch.concat([block1_output, block2_output], -1)
        
        return self.linear(concatenated)