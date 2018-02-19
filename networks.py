import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from utils import setBoundaries

class UNet(nn.Module):
    # UNet for Heat Transport
    def __init__(self, dtype, image_size=32, max_temp=100, num_filters=64):
        super(UNet, self).__init__()

        self.max_temp = max_temp
        self.image_size = image_size
        assert(image_size >= 4)
        assert(np.log2(image_size) % 1 == 0)
        self.num_layers = int(np.log2(image_size))
        self.num_filters = num_filters
        self.dtype = dtype

        self.encoding_layers = nn.ModuleList()
        self.encoding_bns = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.encoding_layers.append(nn.Conv2d(1, num_filters, kernel_size=4, stride=2, padding=1))
            else:
                self.encoding_layers.append(nn.Conv2d(min(2**(i-1),8)*num_filters, min(2**i, 8)*num_filters, kernel_size=4, stride=2, padding=1))
            self.encoding_bns.append(nn.BatchNorm2d(min(2**i*num_filters, 8*num_filters)))

        self.encoded = None
        
        self.decoding_layers = nn.ModuleList()
        self.decoding_bns = nn.ModuleList()
        for i in range(self.num_layers)[::-1]:
            if i == self.num_layers-1:
                self.decoding_layers.append(nn.ConvTranspose2d(min(2**i*num_filters, 8*num_filters), min(2**(i-1)*num_filters, 8*num_filters), kernel_size=4, stride=2, padding=1))
            elif i == 0:
                self.decoding_layers.append(nn.ConvTranspose2d(num_filters*2, 1, kernel_size=4, stride=2, padding=1))
            else:
                self.decoding_layers.append(nn.ConvTranspose2d(min(2**i*num_filters,8*num_filters)*2, min(2**(i-1)*num_filters, 8*num_filters), kernel_size=4, stride=2, padding=1))
            self.decoding_bns.append(nn.BatchNorm2d(min(max(2**(i-1),1)*num_filters, 8*num_filters)))

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
                x = self.decoding_bns[i](self.decoding_layers[i](F.relu(torch.cat((x,xs[-1*i]), dim=1))))
        decoded = x
        output = (decoded + 1)*self.max_temp / 2
        output = output*self.boundary_mask + input*self.center_mask
        return output


class GrowingUNet(UNet):

    def __init__(self, dtype, image_size=32, start_size=4, max_temp=100, num_filters=64):
        super(GrowingUNet, self).__init__(dtype, image_size, max_temp, num_filters)
        self.setSize(start_size)
        self.start_size = start_size
        self.stage_convs = nn.ModuleList()
        self.stage_deconvs = nn.ModuleList()
        size = start_size
        while size < image_size:
            num_channels = int(min(2**(np.log2(image_size)-np.log2(size)-1), 8)*self.num_filters)
            conv = nn.Conv2d(1, num_channels, kernel_size=1)
            deconv = nn.Conv2d(num_channels, 1, kernel_size=1)
            self.stage_convs.append(conv)
            self.stage_deconvs.append(deconv)
            size *= 2


    def setSize(self, size):
        assert (np.log2(size) % 1 == 0)
        assert (size <= self.image_size)
        assert (size >= 4)
        self.size = size
        self.center_mask = torch.zeros(1,1,self.size,self.size)
        self.center_mask[:,:,:,0] = 1
        self.center_mask[:,:,0,:] = 1
        self.center_mask[:,:,:,-1] = 1
        self.center_mask[:,:,-1,:] = 1
        self.center_mask = Variable(self.center_mask).type(self.dtype)
        self.boundary_mask = 1 - self.center_mask.type(self.dtype)


    def forward(self, x):
        if self.size == self.image_size:
            return super(GrowingUNet, self).forward(x)
        else:
            input = x
            # set up convolution to correct num channels
            stage = int((np.log2(self.size) - np.log2(self.start_size)))
            x = F.leaky_relu(self.stage_convs[stage](x), 0.2)
            xs = []
            num_layers = self.num_layers - int((np.log2(self.image_size) - np.log2(self.size)))
            # encoder
            for i in range(len(self.encoding_layers)-num_layers, len(self.encoding_layers)):
                if i == (len(self.encoding_layers)-1):
                    x = self.encoding_layers[i](x)
                else:
                    x = F.leaky_relu(self.encoding_bns[i](self.encoding_layers[i](x)), 0.2)
                xs.append(x)

            # encoded representation is (batch_size, num_filters*8, 1, 1)-dimensional encoding space
            self.encoded = xs.pop(-1)
            
            # decoder
            for i in range(num_layers):
                if i == 0:
                    x = self.decoding_bns[i](self.decoding_layers[i](F.relu(x)))
                else:
                    x = self.decoding_bns[i](self.decoding_layers[i](F.relu(torch.cat((x,xs[-1*i]), dim=1))))

            decoded = F.tanh(self.stage_deconvs[stage](x))
            output = (decoded + 1)*self.max_temp / 2
            output = output*self.boundary_mask + input*self.center_mask
            return output


class VariableLossUNet(UNet):

    def __init__(self, dtype, image_size=32, max_temp=100, num_filters=64):
        super(VariableLossUNet, self).__init__(dtype, image_size, max_temp, num_filters)
        self.stage_deconvs = nn.ModuleList()
        size = 4
        self.num_stages = 0
        while size < image_size:
            num_channels = int(min(2**(np.log2(image_size)-np.log2(size)-1), 8)*self.num_filters)
            deconv = nn.Conv2d(num_channels, 1, kernel_size=1)
            self.stage_deconvs.append(deconv)
            size *= 2
            self.num_stages += 1

        self.masks = [Variable(torch.zeros(1, 1, 2**i, 2**i)).type(self.dtype) for i in range(2,int(np.log2(self.image_size))+1)]
        for mask in self.masks:
            setBoundaries(mask,1,1,1,1)


    def forward(self, inputs):
        input = x = inputs[-1]
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
        outputs = []
        for i in range(len(self.decoding_layers)):
            if i == 0:
                x = self.decoding_bns[i](self.decoding_layers[i](F.relu(x)))
            elif i == len(self.decoding_layers)-1:
                x = F.tanh(self.decoding_layers[i](F.relu(torch.cat((x,xs[0]), dim=1))))
            else:
                x = self.decoding_bns[i](self.decoding_layers[i](F.relu(torch.cat((x,xs[-1*i]), dim=1))))
                decoded = F.tanh(self.stage_deconvs[i-1](x))
                output = (decoded + 1)*self.max_temp / 2
                output = output*(1-self.masks[i-1].type(self.dtype)) + inputs[i-1]*self.masks[i-1]
                outputs.append(output)
        decoded = x
        output = (decoded + 1)*self.max_temp / 2
        output = output*(1-self.masks[-1].type(self.dtype)) + input*self.masks[-1]
        outputs.append(output)
        return outputs
