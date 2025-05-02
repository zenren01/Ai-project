# run
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
  def __init__(self, channels):
    super(ResNetBlock, self).__init__()
    sequence = list()
    sequence+= [
        nn.ReflectionPad2d(1),
        nn.Conv2d(channels,channels,kernel_size = 3, stride = 1,padding = 0, bias = True),
        nn.InstanceNorm2d(channels),
        nn.ReLU(True),
        nn.Dropout(0.5)
    ]
    self.model = nn.Sequential(*sequence)

  def forward(self, x):
    out = x + self.model(x)
    return out