import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import argparse
import numpy as np

from networks import UNet

parser = argparse.ArgumentParser()

# Training
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--growing', action='store_true', help='enables progressive growing during training')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--epoch_size', type=int, default=300, help='number of data points in an epoch')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate, default=2e-4')
parser.add_argument('--experiment', default='run0', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

# Set up CUDA
if opt.cuda and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    if torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    dtype = torch.FloatTensor

# Define physics loss
def PhysicalLoss():
    kernel = Variable(torch.Tensor(np.array([[[[0, 1/4, 0], [1/4, -1, 1/4], [0, 1/4, 0]]]]))).type(dtype)
    def loss(img):
        return F.conv2d(img, kernel).abs().mean()
    return loss

net = UNet(dtype).type(dtype)

physical_loss = PhysicalLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)

## Training loop
data = torch.zeros(opt.batch_size,1,32,32)
for epoch in range(opt.epochs):
    for sample in range(opt.epoch_size):
        data[:,:,:,0] = np.random.uniform(100)
        data[:,:,0,:] = np.random.uniform(100)
        data[:,:,:,-1] = np.random.uniform(100)
        data[:,:,-1,:] = np.random.uniform(100)
        img = Variable(data).type(dtype)
        output = net(img)
        loss = physical_loss(output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opt.epochs, loss.data[0]))

