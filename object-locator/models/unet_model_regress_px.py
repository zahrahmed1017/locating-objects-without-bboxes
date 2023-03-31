__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 3/19/2023
Adapted from: unet_model.py 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Zahra Ahmed"
__version__ = "1.6.0"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary 

from .unet_parts import *
# from unet_parts import *


class UNet_px(nn.Module):
    def __init__(self, n_channels, n_classes,
                 height, width,
                 ultrasmall=True,
                 device=torch.device('cuda')):
        """
        Instantiate a UNet network.
        :param n_channels: Number of input channels (e.g, 3 for RGB)
        :param n_classes: Number of output classes
        :param height: Height of the input images
        :param ultrasmall: If True, the 5 central layers are removed,
                           resulting in a much smaller UNet.
        :param device: Which torch device to use. Default: CUDA (GPU).
        """
        super(UNet_px, self).__init__()

        self.ultrasmall = ultrasmall
        self.device = device

        # With this network depth, there is a minimum image size. 
        # If using ultrasmall network, minimum image size is 64.
        if not self.ultrasmall:
            if height < 256 or width < 256:
                raise ValueError('Minimum input image size is 256x256, got {}x{}'.\
                             format(height, width))

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        if self.ultrasmall:
            self.down3 = down(256, 512, normaliz=False)
            self.up1 = up(768, 128)
            self.up2 = up(256, 64)
            self.up3 = up(128, 64, activ=False)
        else:
            self.down3 = down(256, 512)
            self.down4 = down(512, 512)
            self.down5 = down(512, 512)
            self.down6 = down(512, 512)
            self.down7 = down(512, 512)
            self.down8 = down(512, 512, normaliz=False)
            self.up1 = up(1024, 512)
            self.up2 = up(1024, 512)
            self.up3 = up(1024, 512)
            self.up4 = up(1024, 512)
            self.up5 = up(1024, 256)
            self.up6 = up(512, 128)
            self.up7 = up(256, 64)
            self.up8 = up(128, 64, activ=False)
        self.outc = outconv(64, n_classes*3) # Output 3 channels, [prob,x,y] for each pixel
        # self.out_nonlin = nn.Sigmoid() # Don't non-linearize the output in the CNN, do that when computing loss because we need logits (raw output) for focal loss

        # This layer is not connected anywhere
        # It is only here for backward compatibility
        self.lin = nn.Linear(1, 1, bias=False)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.ultrasmall:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
        else:
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            x8 = self.down7(x7)
            x9 = self.down8(x8)
            x = self.up1(x9, x8)
            x = self.up2(x, x7)
            x = self.up3(x, x6)
            x = self.up4(x, x5)
            x = self.up5(x, x4)
            x = self.up6(x, x3)
            x = self.up7(x, x2)
            x = self.up8(x, x1)
        x = self.outc(x)
        # x = self.out_nonlin(x)

        # Reshape Bx3xHxW to BxHxWx3
        x = x.permute(0,2,3,1).contiguous()

        return x

if __name__ == '__main__':

    model = UNet_px(3, 1,
                    height=64,
                    width=64,
                    known_n_points=None,
                    device=torch.device('cpu'),
                    ultrasmall=True)
    summary(model, (3, 64, 64))

    # Test
    x = torch.rand(1, 3, 64, 64)
    model.eval()
    out = model(x)
    print(out.shape)

"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 11/11/2019 
"""
