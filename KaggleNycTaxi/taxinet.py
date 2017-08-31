import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np

class TaxiNet(nn.Module):
    def __init__(self, input_nodes, learn_rate=0.1, cuda=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_nodes, 10), nn.LeakyReLU(),
            nn.Linear(10, 5), nn.ReLU(),
            nn.Linear(5, 3), nn.ReLU(),
            nn.Linear(3, 1), nn.ReLU())
        if cuda:
            self.model = self.model.cuda()

        self.cuda = cuda
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

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

        for idx, batch in enumerate(np.arange(0, row_count, batch_size)):
            x = df[features].values[batch:batch + batch_size - 1]
            y = df[loss_col].values[batch:batch + batch_size - 1]
            x, y = torch.Tensor(x), torch.Tensor(y)
            if self.cuda:
                yield idx, Variable(x.cuda()), Variable(y.cuda())
            else:
                yield idx, Variable(x), Variable(y)
