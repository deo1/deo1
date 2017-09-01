import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np

class TaxiNet(nn.Module):
    def __init__(self, input_nodes, learn_rate=0.1, cuda=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_nodes, 7), nn.PReLU(7),
            nn.Linear(7, 3), nn.PReLU(3),
            nn.Linear(3, 1), nn.PReLU())

        # initialize weights
        for m in self.model:
            if isinstance(m, torch.nn.modules.linear.Linear):
                nn.init.kaiming_normal(m.weight)
        
        self.cuda = cuda
        self.loss_function = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

        if cuda:
            self.model = self.model.cuda()
            self.loss_function == self.loss_function.cuda()

    def forward(self, x):
        return self.model(x)

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
