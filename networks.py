import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class UNet(nn.Module):
    # UNet for Heat Transport
    def __init__(self, dtype, image_size=32, max_temp=100, num_filters=64):
        super(UNet, self).__init__()

        self.max_temp = max_temp
        assert(image_size > 2)
        num_layers = np.log2(image_size)
        assert(num_layers == int(num_layers))
        num_layers = int(num_layers)

        self.encoding_layers = []
        self.encoding_bns = []
        for i in range(num_layers):
            if i == 0:
                self.encoding_layers.append(nn.Conv2d(1, num_filters, kernel_size=4, stride=2, padding=1))
            else:
                self.encoding_layers.append(nn.Conv2d(min(2**(i-1),8)*num_filters, min(2**i, 8)*num_filters, kernel_size=4, stride=2, padding=1))
            self.encoding_bns.append(nn.BatchNorm2d(min(2**i*num_filters, 8*num_filters)))
           
        self.encoded = None
        
        self.decoding_layers = []
        self.decoding_bns = []
        for i in range(num_layers)[::-1]:
            if i == num_layers-1:
                self.decoding_layers.append(nn.ConvTranspose2d(min(2**i*num_filters, 8*num_filters), min(2**(i-1)*num_filters, 8*num_filters), kernel_size=4, stride=2, padding=1))
            elif i == 0:
                self.decoding_layers.append(nn.ConvTranspose2d(num_filters*2, 1, kernel_size=4, stride=2, padding=1))
            else:
                self.decoding_layers.append(nn.ConvTranspose2d(min(2**i*num_filters,8*num_filters)*2, min(2**(i-1)*num_filters, 8*num_filters), kernel_size=4, stride=2, padding=1))
            self.decoding_bns.append(nn.BatchNorm2d(min(max(2**(i-1),1)*num_filters, 8*num_filters)))

        self.ens = nn.Sequential(*self.encoding_layers)
        self.dns = nn.Sequential(*self.decoding_layers)
        self.ebns = nn.Sequential(*self.encoding_bns)
        self.dbns = nn.Sequential(*self.decoding_bns)

        self.center_mask = torch.zeros(1,1,image_size,image_size)
        self.center_mask[:,:,:,0] = 1
        self.center_mask[:,:,0,:] = 1
        self.center_mask[:,:,:,-1] = 1
        self.center_mask[:,:,-1,:] = 1
        self.center_mask = Variable(self.center_mask).type(dtype)
        self.boundary_mask = 1 - self.center_mask.type(dtype)

    def forward(self, x):
        input = x
        xs = []
        # encoder
        for i in range(len(self.encoding_layers)):
            if i == 0:
                x = F.leaky_relu(self.encoding_layers[i](x), 0.2)
            elif i == len(self.encoding_layers)-1:
                x = self.encoding_layers[i](x)
            else:
                x = F.leaky_relu(self.encoding_bns[i](self.encoding_layers[i](x)), 0.2)
            xs.append(x)

        # encoded representation is (batch_size, num_filters*8, 1, 1)-dimensional encoding space
        self.encoded = xs.pop(-1)
        
        # decoder
        for i in range(len(self.decoding_layers)):
            if i == 0:
                x = self.decoding_bns[i](self.decoding_layers[i](F.relu(x)))
            elif i == len(self.decoding_layers)-1:
                x = F.tanh(self.decoding_layers[i](F.relu(torch.cat((x,xs[0]), dim=1))))
            else:
                x = self.decoding_bns[i](self.decoding_layers[i](F.relu(torch.cat((x,xs[len(self.decoding_layers)-i-1]), dim=1))))
        decoded = x
        output = (decoded + 1)*self.max_temp / 2
        output = output*self.boundary_mask + input*self.center_mask
        return output

