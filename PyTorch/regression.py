# https://github.com/pytorch/examples/blob/master/regression/main.py
from __future__ import print_function
from itertools import count

import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5

def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.4f} x^{} '.format(w, len(W) - i)
    result += '{:+.4f}'.format(b[0])
    return result

def get_batch(batch_size=32, randomize=False):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if randomize: y += torch.randn(1)
    return Variable(x), Variable(y)

class RegressionNet(nn.Module):
    def __init__(self, input_nodes, learn_rate=0.1):
        super().__init__()
        self.hidden = nn.Linear(input_nodes, 1)

        self.loss_function = nn.SmoothL1Loss()
        self.lr = learn_rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learn_rate)

    def forward(self, x):
        return self.hidden(x)

    def learn(self, output, targets):
        loss = self.loss_function(output, targets)

        # Backward pass and weights update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

# Define model
regression_net = RegressionNet(W_target.size(0))

for batch_idx in count(1):
    # Get data
    batch_x, batch_y = get_batch()

    # Forward pass then backward pass
    output = regression_net(batch_x)
    loss = regression_net.learn(output, batch_y)

    # Stop criterion
    if loss.data[0] < 1e-3:
        break

print('Loss: {:.6f} after {} batches'.format(loss.data[0], batch_idx))
print('==> Learned function:\t' + poly_desc(regression_net.hidden.weight.data.view(-1), regression_net.hidden.bias.data))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))