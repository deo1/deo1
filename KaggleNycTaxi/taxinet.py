import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np

class TaxiNet(nn.Module):
    def __init__(self, input_nodes, learn_rate=0.01, cuda=False, max_output=float("inf")):
        super().__init__()
        
        # TODO: batchnorm layers cause bug with cuda=True
        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_nodes, 10),        # affine
            nn.BatchNorm1d(10, momentum=0.05), # normalize mean/variance
            nn.PReLU(10),                      # adative leaky
            nn.Dropout(p=0.002),               # regularize (small dropout when batchnorm)
            
            # Layer 2
            nn.Linear(10, 7),                  # affine
            nn.BatchNorm1d(7, momentum=0.05),  # normalize mean/variance
            nn.PReLU(7),                       # adaptive leaky
            nn.Dropout(p=0.002),               # regularize (small dropout when batchnorm)

            # Layer 3
            nn.Linear(7, 3),                   # affine
            nn.BatchNorm1d(3, momentum=0.05),  # normalize mean/variance
            nn.PReLU(3),                       # adaptive leaky
            nn.Dropout(p=0.002),               # regularize (small dropout when batchnorm)

            # Layer 3
            nn.Linear(3, 1),                   # affine
            nn.ReLU()                          # final output is [0, oo)
        )

        # initialize weights
        for m in self.model:
            if isinstance(m, torch.nn.modules.linear.Linear):
                nn.init.kaiming_normal(m.weight)
        
        self.cuda = cuda
        self.max_output = max_output
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

        if cuda:
            self.model = self.model.cuda()
            self.loss_function == self.loss_function.cuda()

    def forward(self, x):
        return torch.clamp(self.model(x), max=self.max_output)

    def learn(self, output, targets):
        loss = self.loss_function(output, targets)

        # Backward pass and weights update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def get_batches(self, df, loss_col, batch_size=32, exclude=[None]):
        """A generator that returns a training batch (design matrix) in chunks"""

        exclude += [loss_col]
        row_count = df.shape[0]
        features = [col for col in df.columns if col not in exclude]

        # for relatively small batches, it's fine to miss some off the end
        # we'll catch it in the next epoch
        for idx, batch in enumerate(np.arange(0, row_count, batch_size)):
            x = df[features].values[batch:batch + batch_size]
            y = df[loss_col].values[batch:batch + batch_size]
            x, y = torch.Tensor(x), torch.Tensor(y)

            if self.cuda:
                yield idx, Variable(x.cuda()), Variable(y.cuda())
            else:
                yield idx, Variable(x), Variable(y)
