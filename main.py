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

from networks import UNet, GrowingUNet
from solve import solve

import ipdb

parser = argparse.ArgumentParser()

# Training
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--growing', action='store_true', help='enables progressive growing during training')
parser.add_argument('--image_size', type=int, default=32, help='size of image')
parser.add_argument('--start_size', type=int, default=4, help='starting size of image for growing')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--epochs', type=int, default=128, help='number of epochs to train for')
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
    reductions = []
    full_width = opt.image_size
    reduced_width = full_width
    while reduced_width > 32:
        reduced_width /= 4
        indices = np.round(np.linspace(0, full_width-1, reduced_width)).astype(np.int32)
        reductions.append(np.ix_(indices, indices))
    def loss(img):
        loss = F.conv2d(img, kernel).abs().mean()
        for rows, cols in reductions:
            loss += F.conv2d(img[:,:,rows,cols], kernel).abs().mean()
        return loss
    return loss

if not opt.growing:
    opt.start_size = opt.image_size

net = GrowingUNet(dtype, image_size=opt.image_size, start_size=opt.start_size).type(dtype)
print(net)

physical_loss = PhysicalLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)

## Outer training loop
size = opt.start_size
epoch = 0
num_stages = int(np.log2(opt.image_size) - np.log2(opt.start_size)) + 1
stage = 0

while True:
    if num_stages >= 1:
        epochs = int(opt.epochs*2**(-1*(num_stages-stage)))
    else:
        epochs = opt.num_epochs

    fixed_sample_0 = torch.zeros(1,1,size,size)
    fixed_sample_0[:,:,:,0] = 100
    fixed_sample_0[:,:,0,:] = 0
    fixed_sample_0[:,:,:,-1] = 100
    fixed_sample_0[:,:,-1,:] = 0
    fixed_sample_0 = Variable(fixed_sample_0).cuda()

    fixed_sample_1 = torch.zeros(1,1,size,size)
    fixed_sample_1[:,:,:,0] = 100
    fixed_sample_1[:,:,0,:] = 100
    fixed_sample_1[:,:,:,-1] = 100
    fixed_sample_1[:,:,-1,:] = 100
    fixed_sample_1 = Variable(fixed_sample_1).cuda()

    boundary = np.zeros((size, size), dtype=np.bool)
    boundary[0,:] = True
    boundary[-1,:] = True
    boundary[:,0] = True
    boundary[:,-1] = True

    fixed_solution_0 = solve(fixed_sample_0.cpu().data.numpy()[0,0,:,:], boundary, tol=1e-4)
    fixed_solution_1 = solve(fixed_sample_1.cpu().data.numpy()[0,0,:,:], boundary, tol=1e-4)

    ## Inner training loop
    data = torch.zeros(opt.batch_size,1,size,size)
    #data = torch.zeros(opt.batch_size,1,opt.image_size,opt.image_size)
    for _epoch in range(epochs):
        mean_loss = 0
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
            mean_loss += loss.data[0]
        mean_loss /= opt.epoch_size
        print('epoch [{}/{}], size {}, loss:{:.4f}'
              .format(epoch+1, opt.epochs, size, mean_loss))
        epoch += 1

        # Plot real samples
        plt.figure(figsize=(20, 15))
        f_0 = net(fixed_sample_0)
        f_1 = net(fixed_sample_1)
        plt.subplot(2,2,1)
        plt.imshow(f_0.cpu().data.numpy()[0,0,:,:], vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.subplot(2,2,2)
        plt.imshow(f_1.cpu().data.numpy()[0,0,:,:], vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.subplot(2,2,3)
        plt.imshow(fixed_solution_0, vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.subplot(2,2,4)
        plt.imshow(fixed_solution_1, vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.savefig('%s/f_1_epoch%d.png' % (opt.experiment, epoch))
        plt.close()

        # checkpoint networks
        if epoch % 50 == 0:
            torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.experiment, epoch))

        if epoch >= opt.epochs:
            torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.experiment, epoch))
            exit()

    if size < opt.image_size:
        size *= 2
        net.setSize(size)
        stage += 1



