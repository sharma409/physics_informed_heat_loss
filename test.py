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
from utils import setBoundaries, makeSamples, PhysicalLoss
from solve import solve

import ipdb

parser = argparse.ArgumentParser()

from glob import glob

parser.add_argument('--experiment', default='run0', help='folder to output images and model checkpoints')
parser.add_argument('--image_size', type=int, default=32, help='size of image')
parser.add_argument('--growing', action='store_true', help='enables progressive growing during training')
parser.add_argument('--num_test', type=int, default=100, help='size of image')
parser.add_argument('--cuda', action='store_true', help='enables cuda')


opt = parser.parse_args()
print(opt)

assert(os.path.isdir(opt.experiment))

# Set up CUDA
if opt.cuda and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    if torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    dtype = torch.FloatTensor


files = glob(opt.experiment+"/*.pth")
maximum = 0
for file in files:
	maximum = max(int(file.split("_")[-1].split(".")[0]), maximum)
file = glob(opt.experiment + "/*" + str(maximum) + ".pth")[0]
print(file)

if not opt.growing:
	net = UNet(dtype, image_size=opt.image_size).type(dtype)
else:
	net = VariableLossUNet(dtype, image_size=opt.image_size).type(dtype)
state_dict = torch.load(file)
# state_dict = torch.load(file, map_location=lambda storage, loc: storage.cuda(1))
net.load_state_dict(state_dict)
print(net)

physical_loss = PhysicalLoss(dtype)

boundary = np.zeros((opt.image_size, opt.image_size), dtype=np.bool)
boundary[0,:] = True
boundary[-1,:] = True
boundary[:,0] = True
boundary[:,-1] = True

data = torch.zeros(1,1,opt.image_size,opt.image_size)
error = []
for i in range(opt.num_test):
    data[:,:,:,0] = np.random.uniform(100)
    data[:,:,0,:] = np.random.uniform(100)
    data[:,:,:,-1] = np.random.uniform(100)
    data[:,:,-1,:] = np.random.uniform(100)

    solution = solve(data.cpu().numpy()[0,0,:,:], boundary, tol=1e-10)

    img = Variable(data).type(dtype)
    output = net(img)
    loss = physical_loss(output)

    output = output.cpu().data.numpy()[0,0,:,:]

    error.append(np.mean(np.abs(output-solution))) 
    print("%d Error: %.2f, Loss: %.2f" % (i, error[-1], loss.data[0]))
    # Plot real samples
    plt.figure(figsize=(15, 25))
    XX, YY = np.meshgrid(np.arange(0, opt.image_size), np.arange(0, opt.image_size))
    plt.subplot(3,1,1)
    plt.contourf(XX, YY, data.cpu().numpy()[0,0,:,:], colorinterpolation=50, vmin=0, vmax=100, cmap=plt.cm.jet)
    plt.title("Initial Condition")
    plt.axis('equal')
    plt.subplot(3,1,2)
    plt.contourf(XX, YY, solution, colorinterpolation=50, vmin=0, vmax=100, cmap=plt.cm.jet)
    plt.title("Equilibrium Condition")
    plt.axis('equal')
    plt.subplot(3,1,3)
    plt.contourf(XX, YY, output, colorinterpolation=50, vmin=0, vmax=100, cmap=plt.cm.jet)
    plt.title("Learned Output")
    plt.axis('equal')
    plt.savefig('%s/test_%d.png' % (opt.experiment, i))
    plt.close()

error = np.array(error)
print("error: ", np.mean(error))