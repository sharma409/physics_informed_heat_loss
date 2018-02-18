import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import argparse
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from networks import UNet
from utils import makeSamples, plotSamples
from solve import solve

parser = argparse.ArgumentParser()

# Training
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--image_size', type=int, default=32, help='size of image')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--epochs', type=int, default=128, help='number of epochs to train for')
parser.add_argument('--epoch_size', type=int, default=300, help='number of data points in an epoch')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate, default=2e-4')
parser.add_argument('--experiment', default='run0', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--test', action='store_true', help='create test images')
parser.add_argument('--num_test', type=int, default=100, help='size of image')

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

def main():
    net = UNet(dtype, image_size=opt.image_size).type(dtype)
    print(net)

    if opt.test:
        runTest(net)
        return

    physical_loss = PhysicalLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)

    fixed_sample_0, fixed_solution_0, fixed_sample_1, fixed_solution_1 = makeSamples(opt.image_size)

    data = torch.zeros(opt.batch_size,1,opt.image_size,opt.image_size)
    print("Training Started")
    for epoch in range(opt.epochs):
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
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, opt.epochs, mean_loss))

        plotSamples(fixed_solution_0, net(fixed_sample_0), 
                    fixed_solution_1, net(fixed_sample_1),
                    opt.experiment, epoch)

        # checkpoint networks
        if epoch+1 % 50 == 0:
            torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.experiment, epoch+1))

    print("Training Complete")
    torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.experiment, epoch+1))
    print("Network Weights Saved in %s" % opt.experiment)


def runTest(net):
    files = glob(opt.experiment+"/*.pth")
    maximum = 0
    for file in files:
        maximum = max(int(file.split("_")[-1].split(".")[0]), maximum)
    file = glob(opt.experiment + "/*" + str(maximum) + ".pth")[0]
    print(file)

    state_dict = torch.load(file)
    # state_dict = torch.load(file, map_location=lambda storage, loc: storage.cuda(1))
    net.load_state_dict(state_dict)

    physical_loss = PhysicalLoss()

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

        solution = solve(data.cpu().numpy()[0,0,:,:], boundary, tol=1e-5)

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
        plt.imshow(data.cpu().numpy()[0,0,:,:], vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.title("Initial Condition")
        plt.axis('equal')
        plt.subplot(3,1,2)
        plt.imshow(solution, vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.title("Equilibrium Condition")
        plt.axis('equal')
        plt.subplot(3,1,3)
        plt.imshow(output, vmin=0, vmax=100, cmap=plt.cm.jet)
        plt.title("Learned Output")
        plt.axis('equal')
        plt.savefig('%s/test_%d.png' % (opt.experiment, i))
        plt.close()

    error = np.array(error)
    print("error: ", np.mean(error))

if __name__ == '__main__':
    main()