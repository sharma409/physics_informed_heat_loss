import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from solve import solve

def setBoundaries(data, top, bottom, left, right):
    data[:,:,0,1:-1] = top
    data[:,:,-1,1:-1] = bottom
    data[:,:,1:-1,0] = left
    data[:,:,1:-1,-1] = right

    data[:,:,0,0] = (top + left) / 2
    data[:,:,0,-1] = (top + right) / 2
    data[:,:,-1,0] = (bottom + left) / 2
    data[:,:,-1,-1] = (bottom + right) / 2


# Define physics loss
def PhysicalLoss(dtype):
    kernel = Variable(torch.Tensor(np.array([[[[0, 1/4, 0], [1/4, -1, 1/4], [0, 1/4, 0]]]]))).type(dtype)
    def loss(img):
        return F.conv2d(img, kernel).abs().mean()
    return loss


def makeSamples(size):
    fixed_sample_0 = torch.zeros(1,1,size,size)
    setBoundaries(fixed_sample_0, 100,100,0,0)
    fixed_sample_0 = Variable(fixed_sample_0).cuda()

    fixed_sample_1 = torch.zeros(1,1,size,size)
    setBoundaries(fixed_sample_1, 100,100,100,100)
    fixed_sample_1 = Variable(fixed_sample_1).cuda()

    boundary = np.zeros((1,1,size, size), dtype=np.bool)
    setBoundaries(boundary, 1, 1, 1, 1)

    fixed_solution_0 = solve(fixed_sample_0.cpu().data.numpy()[0,0,:,:], np.squeeze(boundary), tol=1e-5)
    fixed_solution_1 = solve(fixed_sample_1.cpu().data.numpy()[0,0,:,:], np.squeeze(boundary), tol=1e-5)

    return fixed_sample_0, fixed_solution_0, fixed_sample_1, fixed_solution_1

def plotSamples(f_0, n_0, f_1, n_1, experiment, epoch):
    plt.figure(figsize=(20, 15))
    plt.subplot(2,2,1)
    plt.imshow(n_0.cpu().data.numpy()[0,0,:,:], vmin=0, vmax=100, cmap=plt.cm.jet)
    plt.axis('equal')
    plt.subplot(2,2,2)
    plt.imshow(n_1.cpu().data.numpy()[0,0,:,:], vmin=0, vmax=100, cmap=plt.cm.jet)
    plt.axis('equal')
    plt.subplot(2,2,3)
    plt.imshow(f_0, vmin=0, vmax=100, cmap=plt.cm.jet)
    plt.axis('equal')
    plt.subplot(2,2,4)
    plt.imshow(f_1, vmin=0, vmax=100, cmap=plt.cm.jet)
    plt.axis('equal')
    plt.savefig('%s/fig_epoch%d.png' % (experiment, epoch))
    plt.close()


def makeGrowingSamples(image_size):
    samps_0 = []
    samps_1 = []
    sols_0 = []
    sols_1 = []
    size = 4
    while size <= image_size:
        fixed_sample_0 = torch.zeros(1,1,size,size)
        setBoundaries(fixed_sample_0, 100,100,0,0)
        fixed_sample_0 = Variable(fixed_sample_0).cuda()

        fixed_sample_1 = torch.zeros(1,1,size,size)
        setBoundaries(fixed_sample_1, 100,100,100,100)
        fixed_sample_1 = Variable(fixed_sample_1).cuda()

        boundary = np.zeros((1,1,size, size), dtype=np.bool)
        setBoundaries(boundary, 1, 1, 1, 1)

        fixed_solution_0 = solve(fixed_sample_0.cpu().data.numpy()[0,0,:,:], np.squeeze(boundary))
        fixed_solution_1 = solve(fixed_sample_1.cpu().data.numpy()[0,0,:,:], np.squeeze(boundary))

        samps_0.append(fixed_sample_0)
        samps_1.append(fixed_sample_1)
        sols_0.append(fixed_solution_0)
        sols_1.append(fixed_solution_1)
        size *= 2

    return samps_0, sols_0, samps_1, sols_1


def boundaryPlot(img):
    r,c = img.shape
    img = np.copy(img)
    img[-20:-1,:] = img[-1,int(c/2.0)]
    img[1:21,:] = img[0,int(c/2.0)]
    img[:,1:21] = img[int(r/2.0), 0]
    img[:,-20:-1] = img[int(r/2.0),-1]
    return img
