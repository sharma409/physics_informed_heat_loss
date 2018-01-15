import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class UNet(nn.Module):
    # UNet for Heat Transport
    def __init__(self, dtype, image_size=32, max_temp=100, num_filters=64):
        super(UNet, self).__init__()

        self.max_temp = max_temp

        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=4, stride=2,padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, padding=1)

        self.ebn1 = nn.BatchNorm2d(num_filters)
        self.ebn2 = nn.BatchNorm2d(num_filters*2)
        self.ebn3 = nn.BatchNorm2d(num_filters*4)
        self.ebn4 = nn.BatchNorm2d(num_filters*8)
        self.ebn5 = nn.BatchNorm2d(num_filters*8)

        self.encoded = None # 4x4x4
        
        self.deconv1 = nn.ConvTranspose2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(num_filters*8*2, num_filters*4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(num_filters*4*2, num_filters*2, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(num_filters*2*2, num_filters, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(num_filters*2, 1, kernel_size=4, stride=2, padding=1)

        self.dbn1 = nn.BatchNorm2d(num_filters*8)
        self.dbn2 = nn.BatchNorm2d(num_filters*4)
        self.dbn3 = nn.BatchNorm2d(num_filters*2)
        self.dbn4 = nn.BatchNorm2d(num_filters)

        self.center_mask = torch.zeros(1,32,32).unsqueeze(0)
        self.center_mask[:,:,:,0] = 1
        self.center_mask[:,:,0,:] = 1
        self.center_mask[:,:,:,-1] = 1
        self.center_mask[:,:,-1,:] = 1
        self.center_mask = Variable(self.center_mask).type(dtype)
        self.boundary_mask = 1 - self.center_mask.type(dtype)

    def forward(self, x):
        # encoder
        en_1 = F.leaky_relu(self.conv1(x), 0.2)
        en_2 = F.leaky_relu(self.ebn2(self.conv2(en_1)), 0.2)
        en_3 = F.leaky_relu(self.ebn3(self.conv3(en_2)), 0.2)
        en_4 = F.leaky_relu(self.ebn4(self.conv4(en_3)), 0.2)

        # encoded representation is (batch_size, num_filters*8, 1, 1)-dimensional encoding space
        self.encoded = self.conv5(en_4)
        
        # decoder
        de_1 = self.dbn1(self.deconv1(F.relu(self.encoded)))
        de_2 = self.dbn2(self.deconv2(F.relu(torch.cat((de_1, en_4), dim=1))))
        de_3 = self.dbn3(self.deconv3(F.relu(torch.cat((de_2, en_3), dim=1))))
        de_4 = self.dbn4(self.deconv4(F.relu(torch.cat((de_3, en_2), dim=1))))
        decoded = F.tanh(self.deconv5(F.relu(torch.cat((de_4, en_1), dim=1))))
        output = (decoded + 1)*self.max_temp / 2
        output = output*self.boundary_mask + x*self.center_mask
        return output

