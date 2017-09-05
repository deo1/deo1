import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np

class TaxiNet(nn.Module):
    def __init__(self, input_nodes, learn_rate=0.01, cuda=False, max_output=float("inf")):
        super().__init__()
        
        # TODO: batchnorm layers cause bug with cuda=True
        # TODO: convolutional layers on the coordinates
        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_nodes, 100),        # affine
            nn.BatchNorm1d(100, momentum=0.05), # normalize mean/variance
            nn.PReLU(100),                      # adative leaky
            nn.Dropout(p=0.002),                # regularize (small dropout when batchnorm)
            
            # Layer 2
            nn.Linear(100, 50),                 # affine
            nn.BatchNorm1d(50, momentum=0.05),  # normalize
            nn.PReLU(50),                       # adaptive leaky
            nn.Dropout(p=0.002),                # regularize

            # Layer 3
            nn.Linear(50, 5),                   # affine
            nn.BatchNorm1d(5, momentum=0.05),   # normalize
            nn.PReLU(5),                        # adaptive leaky
            nn.Dropout(p=0.002),                # regularize

            # Layer 3
            nn.Linear(5, 1),                    # affine
            nn.ReLU()                           # final output is [0, oo)
        )

        # initialize weights
        for m in self.model:
            if isinstance(m, torch.nn.modules.linear.Linear):
                nn.init.kaiming_normal(m.weight)
        
        self.cuda = cuda
        self.max_output = max_output
        self.loss_function = nn.MSELoss()
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

    def learn_loop(self, data, loss_column, epochs, batch_size, exclude = [], randomize=True, chatty=1):
        self.train() # train mode (learn batchnorm mean/var)
        for epoch in range(epochs):
            for batch_idx, batch_x, batch_y in \
                self.get_batches(data, loss_column, batch_size=batch_size, exclude=exclude):
                
                # Forward pass then backward pass
                # TODO : cross validation
                output = self.forward(batch_x)
                loss = self.learn(output, batch_y)
                if chatty > 0: self.print_minibatch_loop(loss.data[0]**0.5, output, batch_idx, batch_size, data.shape[0], epoch)
    
            # score and train on the whole set to see where we're at
            _, all_x, all_y = next(self.get_batches(data, loss_column, batch_size=data.shape[0], exclude=exclude))
            train_y = self(all_x)
            train_loss = self.loss_function(train_y, all_y)
            if chatty > 0:
                print('\nLoss: {:.4f} after {} epochs'.format(train_loss.data[0]**0.5, epoch))
            
            # shuffle the data so that new batches / orders are used in the next epoch
            if randomize: data = data.sample(frac=1).reset_index(drop=True)


    # TODO
    def save_model(self):
        pass

    @staticmethod
    def print_minibatch_loop(loss, output, batch_idx, batch_size, total_samples, epoch):
        print('\rLoss: {:.4f} after {} batches ({:.1f}%), {} epochs. (med(y): {:.1f}){}'.format(
            loss,                                          # iteration loss
            batch_idx,                                     # iteration count
            100 * batch_idx * batch_size / total_samples,  # % complete within epoch
            epoch,                                         # epoch count
            output.median().data[0],                       # to monitor that the weights haven't saturated to 0
            "       "), end="")
