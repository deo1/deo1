import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split

class ModulePlus(nn.Module):
    """
    A generic module with many helpers built in for batching and optimizing.
    """

    def __init__(self, learn_rate=0.01, cuda=False):
        super().__init__()         
        self.cuda = cuda
        self.learn_rate = learn_rate
        self.train_loss = []
        self.train_cv_loss = []
        self.best_cv_loss = None
        self.state_dicts = []

        if cuda:
            self.model = self.model.cuda()
            self.loss_function == self.loss_function.cuda()

    def learn(self, output, targets):
        loss = self.loss_function(output, targets)**0.5

        # Backward pass and weights update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def get_batches(self, df, loss_col, batch_size=32, exclude=[None], cv=False):
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

            if self.cuda and not cv:
                yield idx, Variable(x.cuda()), Variable(y.cuda())
            elif self.cuda and cv:
                yield idx, Variable(x.cuda(), volatile=True), Variable(y.cuda(), volatile=True)
            elif not cv:
                yield idx, Variable(x), Variable(y)
            else:
                yield idx, Variable(x, volatile=True), Variable(y, volatile=True)

    def learn_loop(self, data, loss_column, epochs, batch_size, exclude=[],
                   lr_decay_factor=0.1, lr_decay_epoch=10, cv = 0.0,
                   early_stopping_rounds=None, randomize=True, chatty=1):

        self.train() # train mode (learn batchnorm mean/var)
        if early_stopping_rounds == None: early_stopping_rounds = epochs
        if cv > 0.0:
            train, train_cv = train_test_split(data, test_size=cv)
        else:
            train = data
            train_cv = data

        for epoch in range(epochs):
            
            # lower the learning rate as we progress
            if lr_decay_factor < 1:
                self.lr_scheduler(epoch, lr_decay_factor, lr_decay_epoch)

            for batch_idx, batch_x, batch_y in \
                self.get_batches(train, loss_column, batch_size=batch_size, exclude=exclude):
                
                # Forward pass then backward pass
                output = self.forward(batch_x)
                loss = self.learn(output, batch_y)
                if chatty > 0: self.print_minibatch_loop(loss.data[0], output, batch_idx, batch_size, train.shape[0], epoch)
    
            # score and train on the whole set to see where we're at
            stop = self.early_stopping_rounds(train, train_cv, early_stopping_rounds, loss_column, exclude)
            if stop: return epoch

            # shuffle the data so that new batches / orders are used in the next epoch
            if randomize: train = train.sample(frac=1).reset_index(drop=True)

    def early_stopping_rounds(self, train, train_cv, early_stopping_rounds,
                              loss_column, exclude, chatty=1):

            _, train_x, train_y = next(self.get_batches(train, loss_column, batch_size=train.shape[0], exclude=exclude, cv=True))
            _, train_cv_x, train_cv_y = next(self.get_batches(train_cv, loss_column, batch_size=train_cv.shape[0], exclude=exclude, cv=True))
            out_y = self(train_x)
            out_cv_y = self(train_cv_x)
            self.train_loss.append((self.loss_function(out_y, train_y)**0.5).data[0])
            self.train_cv_loss.append((self.loss_function(out_cv_y, train_cv_y)**0.5).data[0])
            self.state_dicts.append(self.state_dict())
            if chatty > 0:
                print('\nTrain Loss: {:.4f}, CV Loss: {:.4f} ({:.4f}) after {} epochs'.format(
                    self.train_loss[-1],
                    self.train_cv_loss[-1],
                    self.train_loss[-1] - self.train_cv_loss[-1],
                    len(self.train_cv_loss)))
            
            self.best_cv_loss = min(self.train_cv_loss)

            if (len(self.train_cv_loss) >= early_stopping_rounds and
                self.train_cv_loss[-early_stopping_rounds] < self.train_cv_loss[-1]):
                print('Early stopping') 
                best_model = self.state_dicts[self.train_cv_loss.index(self.best_cv_loss)]
                self.load_state_dict(best_model)
                return True
            else:
                return False

    def lr_scheduler(self, epoch, factor=0.1, lr_decay_epoch=10):
        """Decay learning rate by a factor of `factor` every `lr_decay_epoch` epochs."""

        lr = self.learn_rate * (factor**(epoch // lr_decay_epoch))
    
        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))
    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

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
