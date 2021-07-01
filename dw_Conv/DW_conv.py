import torch.nn as nn
from torchsummary import summary


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, kernels_per_layer=1):
      super(depthwise_separable_conv, self).__init__()
      self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=1, groups=nin)
      self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
      out = self.depthwise(x)
      out = self.pointwise(out)
      return out
