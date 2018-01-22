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

from networks import UNet, GrowingUNet, VariableLossUNet
from utils import setBoundaries, makeSamples

import ipdb

parser = argparse.ArgumentParser()

# Training
parser.add_argument('--cuda', action='store_true', help='enables cuda')
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
    def loss(img):
        return F.conv2d(img, kernel).abs().mean()
    return loss

net = VariableLossUNet(dtype, image_size=opt.image_size).type(dtype)
print(net)

physical_loss = PhysicalLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)

## Outer training loop
size = opt.start_size
epoch = 0
num_stages = int(np.log2(opt.image_size) - np.log2(opt.start_size)) + 1
num_losses = int(np.log2(opt.image_size) - 2) + 1
loss_weights = np.zeros(num_losses)
loss_weights[int(np.log2(opt.start_size) - 2)] = 1
for i in range(num_losses):
    print ("loss weight: ", loss_weights[i], "image size: ", 4*2**i)
stage = 0

samps_0, sols_0, samps_1, sols_1 = makeSamples(opt.image_size)

def weighted_loss(outputs, loss_weights):
    loss = 0
    for i in range(len(outputs)):
        loss += physical_loss(outputs[i])*loss_weights[i]
    return loss

while True:
    if num_stages >= 1:
        epochs = int(opt.epochs*2**(-1*(num_stages-stage)))
    else:
        epochs = opt.num_epochs

    ## Inner training loop
    data = [Variable(torch.zeros(opt.batch_size, 1, 2**j, 2**j)).type(dtype) for j in range(2,int(np.log2(opt.image_size))+1)]
    for _epoch in range(epochs):
        for sample in range(opt.epoch_size):
            top = np.random.uniform(100)
            bottom = np.random.uniform(100)
            left = np.random.uniform(100)
            right = np.random.uniform(100)
            for k, _ in enumerate(data):
                setBoundaries(data[k], top, bottom, left, right)

            outputs = net(data)
            loss = weighted_loss(outputs, loss_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch [{}/{}], size {}, loss:{:.4f}'
              .format(epoch+1, opt.epochs, size, loss.data[0]))
        epoch += 1

        # Plot real samples
        plt.figure(figsize=(20, 15))
        f_0 = net(samps_0)[stage]
        f_1 = net(samps_1)[stage]
        XX, YY = np.meshgrid(np.arange(0, size), np.arange(0, size))
        plt.subplot(2,2,1)
        plt.contourf(XX, YY, f_0.cpu().data.numpy()[0,0,:,:], colorinterpolation=50, vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.subplot(2,2,2)
        plt.contourf(XX, YY, f_1.cpu().data.numpy()[0,0,:,:], colorinterpolation=50, vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.subplot(2,2,3)
        plt.contourf(XX, YY, sols_0[stage], colorinterpolation=50, vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.subplot(2,2,4)
        plt.contourf(XX, YY, sols_1[stage], colorinterpolation=50, vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.savefig('%s/f_1_epoch%d.png' % (opt.experiment, epoch))
        plt.close()

        # checkpoint networks
        if epoch % 5 == 0:
            torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.experiment, epoch))

        if epoch >= opt.epochs:
            exit()

    if loss_weights[-1] == 0:
        size *= 2
        loss_weights = np.roll(loss_weights, 1)
        for i in range(num_losses):
            print ("loss weight: ", loss_weights[i], "image size: ", 4*2**i)
        stage += 1

