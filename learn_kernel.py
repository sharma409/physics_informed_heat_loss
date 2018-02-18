import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from solve import solve
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--size', type=int, default=32, help='size of image')

opt = parser.parse_args()
print(opt)

# Set up CUDA
if opt.cuda and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    if torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    dtype = torch.FloatTensor

class kernel(nn.Module):
    # UNet for Heat Transport
    def __init__(self):
        super(kernel, self).__init__()
        self.kernel = nn.Conv2d(1, 1, kernel_size=3, bias=False)

    def forward(self, x):
        return self.kernel(x)

def PhysicalLoss():
    kernel = Variable(torch.Tensor(np.array([[[[0, 1/4, 0], [1/4, -1, 1/4], [0, 1/4, 0]]]]))).type(dtype)
    def loss(img):
        return F.conv2d(img, kernel).abs().mean()
    return loss

net = kernel().type(dtype)

learning_rate = 2e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
physical_loss = PhysicalLoss()

boundary = np.zeros((opt.size, opt.size), dtype=np.bool)
boundary[0,:] = True
boundary[-1,:] = True
boundary[:,0] = True
boundary[:,-1] = True
data = torch.zeros(1,1,opt.size,opt.size)
i = 0
while True:
    i += 1
    print("Kernel: ", net.state_dict())
    data[:,:,:,0] = np.random.uniform(100)
    data[:,:,0,:] = np.random.uniform(100)
    data[:,:,:,-1] = np.random.uniform(100)
    data[:,:,-1,:] = np.random.uniform(100)
    solution = Variable(torch.Tensor(solve(data.cpu().numpy()[0,0,:,:], boundary, tol=1e-10)).unsqueeze(0).unsqueeze(0)).type(dtype)
    output = net(solution).abs().mean()
    print("Loss: ", output)
    print("Physical Loss: ", physical_loss(solution))
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
    
    

