import torch
from torch.autograd import Variable
import numpy as np

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


def makeSamples(image_size):
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