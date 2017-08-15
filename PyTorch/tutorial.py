''' http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py '''
from __future__ import print_function
import torch

# 5 x 3 matrix (uninitialized)
x = torch.Tensor(5, 3)
print(x)

# 5 x 3 matrix (randomly initialized)
x = torch.Tensor(torch.rand(5, 3))
print(x)

y = torch.rand(5, 3)

print(x[0,0] * y[0,0] == (x * y)[0,0])

a = torch.ones(5)
b = a.numpy()
a.add_(1)
print(list(a) == list(b))

# move tensors to the gpu
if torch.cuda.is_available():
    x2 = x.cuda()
    y2 = y.cuda()
    print(x + y)


''' http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html '''
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

out.backward()
print(x.grad) # gradient d(out) / dx

x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
    
print(y)


''' http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html '''
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # max pooling over a 2x2 window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # square x can only specify single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight

# input to the forward() is an autograd.Variable, and so is the output.
invar = Variable(torch.randn(1, 1, 32, 32)) # a 32x32 pixel image (nSamples x nChannels x Height x Width)
out = net(invar)
print(out)

# zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

out = net(invar)
target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
# etc.

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
