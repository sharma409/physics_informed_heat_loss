import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from networks import UNet
from solve import solve

parser = argparse.ArgumentParser()

# Training
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--growing', action='store_true', help='enables progressive growing during training')
parser.add_argument('--image_size', type=int, default=32, help='size of image')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--epoch_size', type=int, default=300, help='number of data points in an epoch')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate, default=2e-4')
parser.add_argument('--experiment', default='run0', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

# Make output directory
os.makedirs(opt.experiment, exist_ok=True)
with open(os.path.join(opt.experiment, 'config.txt'), 'w') as f:
    f.write(str(opt))

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

net = UNet(dtype, opt.image_size).type(dtype)
print(net)

physical_loss = PhysicalLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)

fixed_sample_0 = torch.zeros(1,1,opt.image_size,opt.image_size)
fixed_sample_0[:,:,:,0] = 100
fixed_sample_0[:,:,0,:] = 0
fixed_sample_0[:,:,:,-1] = 100
fixed_sample_0[:,:,-1,:] = 0
fixed_sample_0 = Variable(fixed_sample_0).cuda()

fixed_sample_1 = torch.zeros(1,1,opt.image_size,opt.image_size)
fixed_sample_1[:,:,:,0] = 100
fixed_sample_1[:,:,0,:] = 100
fixed_sample_1[:,:,:,-1] = 100
fixed_sample_1[:,:,-1,:] = 100
fixed_sample_1 = Variable(fixed_sample_1).cuda()

boundary = np.zeros((opt.image_size, opt.image_size), dtype=np.bool)
boundary[0,:] = True
boundary[-1,:] = True
boundary[:,0] = True
boundary[:,-1] = True

fixed_solution_0 = solve(fixed_sample_0.cpu().data.numpy()[0,0,:,:], boundary)
fixed_solution_1 = solve(fixed_sample_1.cpu().data.numpy()[0,0,:,:], boundary)

## Training loop
data = torch.zeros(opt.batch_size,1,opt.image_size,opt.image_size)
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

    # Plot real samples
    plt.figure(figsize=(20, 15))
    f_0 = net(fixed_sample_0)
    f_1 = net(fixed_sample_1)
    XX, YY = np.meshgrid(np.arange(0, opt.image_size), np.arange(0, opt.image_size))
    plt.subplot(2,2,1)
    plt.contourf(XX, YY, f_0.cpu().data.numpy()[0,0,:,:], colorinterpolation=50, cmap=plt.cm.jet)
    plt.axis('equal')
    plt.subplot(2,2,2)
    plt.contourf(XX, YY, f_1.cpu().data.numpy()[0,0,:,:], colorinterpolation=50, cmap=plt.cm.jet)
    plt.axis('equal')
    plt.subplot(2,2,3)
    plt.contourf(XX, YY, fixed_solution_0, colorinterpolation=50, cmap=plt.cm.jet)
    plt.axis('equal')
    plt.subplot(2,2,4)
    plt.contourf(XX, YY, fixed_solution_1, colorinterpolation=50, cmap=plt.cm.jet)
    plt.axis('equal')
    plt.savefig('%s/f_1_epoch%d.png' % (opt.experiment, epoch))
    plt.close()

    # checkpoint networks
    if epoch % 5 == 0:
        torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.experiment, epoch))


